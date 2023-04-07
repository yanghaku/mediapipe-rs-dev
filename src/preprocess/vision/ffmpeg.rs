use super::*;
use common::ffmpeg_input::FFMpegInput;

pub type FFMpegVideoData = FFMpegInput<ffmpeg_next::decoder::Video, ffmpeg_next::frame::Video>;

impl<'model> InToTensorsIterator<'model> for FFMpegVideoData {
    type Iter = FFMpegVideoToTensorIter<'model>;

    fn into_tensors_iter(self, to_tensor_info: &'model ToTensorInfo) -> Result<Self::Iter, Error> {
        let image_to_tensor_info = to_tensor_info.try_to_image()?;
        let target_format = match image_to_tensor_info.color_space {
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                ffmpeg_next::format::Pixel::RGB24
            }
            ImageColorSpaceType::GRAYSCALE => ffmpeg_next::format::Pixel::GRAY8,
        };
        let scale = if self.decoder.format() != target_format
            || self.decoder.width() != image_to_tensor_info.width
            || self.decoder.height() != image_to_tensor_info.height
        {
            Some((
                ffmpeg_next::software::scaling::Context::get(
                    self.decoder.format(),
                    self.decoder.width(),
                    self.decoder.height(),
                    target_format,
                    image_to_tensor_info.width,
                    image_to_tensor_info.height,
                    ffmpeg_next::software::scaling::Flags::BITEXACT
                        | ffmpeg_next::software::scaling::Flags::SPLINE,
                )?,
                ffmpeg_next::frame::Video::empty(),
            ))
        } else {
            None
        };
        let convert_to_ms = self.decoder.time_base().numerator() as f64
            / self.decoder.time_base().denominator() as f64
            * 1000.;
        Ok(Self::Iter {
            image_to_tensor_info,
            source: self,
            scale,
            convert_to_ms,
        })
    }
}

#[doc(hidden)]
pub struct FFMpegVideoToTensorIter<'a> {
    image_to_tensor_info: &'a ImageToTensorInfo,
    source: FFMpegVideoData,
    scale: Option<(
        ffmpeg_next::software::scaling::Context,
        ffmpeg_next::frame::Video,
    )>,
    convert_to_ms: f64,
}

impl<'a> TensorsIterator for FFMpegVideoToTensorIter<'a> {
    fn next_tensors(
        &mut self,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<Option<u64>, Error> {
        let output_buffer = if output_buffers.as_mut().len() != 1 {
            return Err(Error::ArgumentError(format!(
                "Expect output buffer's shape is `1`, but got `{}`",
                output_buffers.as_mut().len()
            )));
        } else {
            output_buffers.as_mut().get_mut(0).unwrap()
        };

        if !self.source.receive_frame()? {
            return Ok(None);
        }
        let frame = if self.scale.is_some() {
            let (scale, out_frame) = self.scale.as_mut().unwrap();
            // scale frame
            scale.run(&self.source.frame, out_frame)?;
            out_frame
        } else {
            &self.source.frame
        };

        match self.image_to_tensor_info.color_space {
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                let data = frame.data(0);
                let img = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
                    self.image_to_tensor_info.width,
                    self.image_to_tensor_info.height,
                    data,
                )
                .unwrap();
                image::rgb8_image_buffer_to_tensor(&img, self.image_to_tensor_info, output_buffer)?;
            }
            ImageColorSpaceType::GRAYSCALE => {
                todo!("gray image")
            }
        }

        Ok(Some(
            self.source
                .frame
                .pts()
                .map(|t| (t as f64 * self.convert_to_ms) as u64)
                .unwrap_or(0) as u64,
        ))
    }
}
