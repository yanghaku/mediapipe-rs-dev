use super::*;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

type FFMpegVideoInput =
    common::ffmpeg_input::FFMpegInput<ffmpeg_next::decoder::Video, ffmpeg_next::frame::Video>;

pub struct FFMpegVideoData {
    source: FFMpegVideoInput,
    convert_to_ms: f64,

    // mutable caches
    scales: RefCell<HashMap<ScaleKey, ffmpeg_next::software::scaling::Context>>,
    scale_frame_buffer: RefCell<ffmpeg_next::frame::Video>,
}

impl FFMpegVideoData {
    #[inline(always)]
    pub fn new(input: ffmpeg_next::format::context::Input) -> Result<Self, Error> {
        let source = FFMpegVideoInput::new(input)?;
        let convert_to_ms = source.decoder.time_base().numerator() as f64
            / source.decoder.time_base().denominator() as f64
            * 1000.;
        Ok(Self {
            source,
            convert_to_ms,
            scales: RefCell::new(Default::default()),
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
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        let scale_key = ScaleKey {
            src_w: self.0.source.frame.width(),
            src_h: self.0.source.frame.height(),
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
                scale_ctx.run(&self.0.source.frame, &mut scale_frame_buffer)?;
                &*scale_frame_buffer
            } else {
                &self.0.source.frame
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
