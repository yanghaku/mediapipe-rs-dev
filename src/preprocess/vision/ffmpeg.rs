use super::*;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

type FFMpegVideoInput =
    common::ffmpeg_input::FFMpegInput<ffmpeg_next::decoder::Video, ffmpeg_next::frame::Video>;

pub struct FFMpegVideoData {
    source: FFMpegVideoInput,

    // immutable caches
    filter_desc_in: String,
    convert_to_ms: f64,

    // mutable caches
    scales: RefCell<HashMap<ScaleKey, ffmpeg_next::software::scaling::Context>>,
    crop_frame_buffer: RefCell<ffmpeg_next::frame::Video>,
    scale_frame_buffer: RefCell<ffmpeg_next::frame::Video>,
}

impl FFMpegVideoData {
    #[inline(always)]
    pub fn new(input: ffmpeg_next::format::context::Input) -> Result<Self, Error> {
        let source = FFMpegVideoInput::new(input)?;
        let convert_to_ms = source.decoder.time_base().numerator() as f64
            / source.decoder.time_base().denominator() as f64
            * 1000.;
        let filter_desc_in = format!(
            "buffer=video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
            source.decoder.width(),
            source.decoder.height(),
            ffmpeg_next::ffi::AVPixelFormat::from(source.decoder.format()) as u32,
            source.decoder.time_base().numerator(),
            source.decoder.time_base().denominator(),
            source.decoder.aspect_ratio().numerator(),
            source.decoder.aspect_ratio().denominator()
        );
        Ok(Self {
            source,
            filter_desc_in,
            convert_to_ms,
            scales: RefCell::new(Default::default()),
            crop_frame_buffer: RefCell::new(ffmpeg_next::frame::Video::empty()),
            scale_frame_buffer: RefCell::new(ffmpeg_next::frame::Video::empty()),
        })
    }
}

impl VideoData for FFMpegVideoData {
    type Frame<'frame> = FFMpegFrame<'frame>;

    #[inline(always)]
    fn next_frame(&mut self) -> Result<Option<Self::Frame<'_>>, Error> {
        if !self.source.receive_frame()? {
            return Ok(None);
        }
        Ok(Some(FFMpegFrame(self)))
    }
}

pub struct FFMpegFrame<'a>(&'a mut FFMpegVideoData);

impl<'a> ImageToTensor for FFMpegFrame<'a> {
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        to_tensor_info: &ImageToTensorInfo,
        process_options: &ImageProcessingOptions,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        let mut src_width = self.0.source.frame.width();
        let mut src_height = self.0.source.frame.height();

        // crop and rotate
        let mut crop_frame_buffer = self.0.crop_frame_buffer.borrow_mut();
        let src_frame = if process_options.rotation_degrees != 0
            || process_options.region_of_interest.is_some()
        {
            const IN_NODE: &'static str = "Parsed_buffer_0";
            const OUT_NODE_PREFIX: &'static str = "Parsed_buffersink_";
            let mut num_node = 1;

            // config filter desc
            let mut desc = if let Some(ref roi) = process_options.region_of_interest {
                let crop_x = (src_width as f32 * roi.left) as u32;
                let crop_y = (src_height as f32 * roi.top) as u32;
                src_width = (src_width as f32 * (roi.right - roi.left)) as u32;
                src_height = (src_height as f32 * (roi.bottom - roi.top)) as u32;
                num_node += 1;
                format!(
                    "[in];[in]crop=w={}:h={}:x={}:y={}",
                    src_width, src_height, crop_x, crop_y
                )
            } else {
                String::new()
            };
            if process_options.rotation_degrees != 0 {
                num_node += 1;
                desc.extend(
                    format!(
                        "[last_out];[last_out]rotate={}*PI/180",
                        process_options.rotation_degrees
                    )
                    .chars(),
                );
                if process_options.rotation_degrees == 180 {
                    std::mem::swap(&mut src_height, &mut src_width);
                }
            }

            let mut filter_graph = ffmpeg_next::filter::Graph::new();
            filter_graph.parse(
                format!("{}{}[out];[out]buffersink", self.0.filter_desc_in, desc).as_str(),
            )?;
            filter_graph.validate()?;
            filter_graph
                .get(IN_NODE)
                .unwrap()
                .source()
                .add(&self.0.source.frame)?;
            filter_graph
                .get(format!("{}{}", OUT_NODE_PREFIX, num_node).as_str())
                .unwrap()
                .sink()
                .frame(&mut crop_frame_buffer)?;
            &*crop_frame_buffer
        } else {
            &self.0.source.frame
        };

        // scale and convert format
        let scale_key = ScaleKey {
            src_w: src_width,
            src_h: src_height,
            dst_w: to_tensor_info.width,
            dst_h: to_tensor_info.height,
            dst_format: to_tensor_info.color_space,
        };
        let src_format = self.0.source.frame.format();
        let mut scales_cache = self.0.scales.borrow_mut();
        let mut scale_frame_buffer = self.0.scale_frame_buffer.borrow_mut();
        let frame =
            if let Some(scale_ctx) = cached_scale_ctx(&mut scales_cache, src_format, scale_key) {
                // scale frame
                scale_ctx.run(&src_frame, &mut scale_frame_buffer)?;
                &*scale_frame_buffer
            } else {
                &src_frame
            };

        match to_tensor_info.color_space {
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                let data = frame.data(0);
                let img = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
                    to_tensor_info.width,
                    to_tensor_info.height,
                    data,
                )
                .unwrap();
                image::rgb8_image_buffer_to_tensor(&img, to_tensor_info, output_buffer)?;
            }
            ImageColorSpaceType::GRAYSCALE => {
                todo!("gray image")
            }
        }

        Ok(())
    }

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32) {
        (
            self.0.source.decoder.width(),
            self.0.source.decoder.height(),
        )
    }

    /// return the current timestamp (ms)
    fn time_stamp_ms(&self) -> Option<u64> {
        self.0
            .source
            .frame
            .timestamp()
            .map(|t| (t as f64 * self.0.convert_to_ms) as u64)
    }
}

#[derive(Hash, Eq, PartialEq)]
struct ScaleKey {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    dst_format: ImageColorSpaceType,
}

// get cached scale context
fn cached_scale_ctx(
    scales: &mut HashMap<ScaleKey, ffmpeg_next::software::scaling::Context>,
    src_format: ffmpeg_next::format::Pixel,
    key: ScaleKey,
) -> Option<&mut ffmpeg_next::software::scaling::Context> {
    // do not need to scale
    if key.src_w == key.dst_w
        && key.src_h == key.dst_h
        && match key.dst_format {
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                src_format == ffmpeg_next::format::Pixel::RGB24
            }
            ImageColorSpaceType::GRAYSCALE => src_format == ffmpeg_next::format::Pixel::GRAY8,
        }
    {
        return None;
    }

    Some(match scales.entry(key) {
        Entry::Occupied(s) => s.into_mut(),
        Entry::Vacant(v) => {
            // new scale context
            let key = v.key();
            let dst_format = match key.dst_format {
                ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                    ffmpeg_next::format::Pixel::RGB24
                }
                ImageColorSpaceType::GRAYSCALE => ffmpeg_next::format::Pixel::GRAY8,
            };
            let scale = ffmpeg_next::software::scaling::Context::get(
                src_format,
                key.src_w,
                key.src_h,
                dst_format,
                key.dst_w,
                key.dst_h,
                ffmpeg_next::software::scaling::Flags::BITEXACT
                    | ffmpeg_next::software::scaling::Flags::SPLINE,
            )
            .unwrap();
            v.insert(scale)
        }
    })
}
