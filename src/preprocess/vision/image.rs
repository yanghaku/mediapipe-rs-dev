extern crate image as image_crate;

use super::*;
pub(super) use image_crate::{imageops, DynamicImage, EncodableLayout, ImageBuffer, Rgb, RgbImage};

const IMAGE_RESIZE_FILTER: imageops::FilterType = imageops::FilterType::Gaussian;

macro_rules! get_rgb_mean_std_from_info {
    ( $info:ident ) => {{
        let r_mean = $info.normalization_options.0.get(0).unwrap();
        let r_std = $info.normalization_options.1.get(0).unwrap();
        let g_mean = $info.normalization_options.0.get(1).unwrap_or(r_mean);
        let g_std = $info.normalization_options.1.get(1).unwrap_or(r_std);
        let b_mean = $info.normalization_options.0.get(1).unwrap_or(r_mean);
        let b_std = $info.normalization_options.1.get(1).unwrap_or(r_std);
        (r_mean, r_std, g_mean, g_std, b_mean, b_std)
    }};
}

impl ImageToTensor for DynamicImage {
    #[inline]
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        info: &ImageToTensorInfo,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        // need resize
        if info.width != self.width() || info.height != self.height() {
            dynamic_image_into_tensor(
                self.resize_exact(info.width, info.height, IMAGE_RESIZE_FILTER),
                info,
                output_buffer,
            )
        } else {
            match info.color_space {
                ImageColorSpaceType::GRAYSCALE => {
                    unimplemented!()
                }
                // we treat unknown as rgb8
                ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                    if let Some(rgb) = self.as_rgb8() {
                        rgb8_image_buffer_to_tensor(rgb, info, output_buffer)
                    } else {
                        rgb8_image_buffer_to_tensor(&self.to_rgb8(), info, output_buffer)
                    }
                }
            }
        }
    }

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }
}

impl ImageToTensor for RgbImage {
    #[inline]
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        info: &ImageToTensorInfo,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        if info.width != self.width()
            || info.height != self.height()
            || info.color_space == ImageColorSpaceType::GRAYSCALE
        {
            // must resize or change to gray
            let mut dynamic_img = DynamicImage::from(self.clone());
            if info.width != self.width() || info.height != self.height() {
                dynamic_img =
                    dynamic_img.resize_exact(info.width, info.height, IMAGE_RESIZE_FILTER);
            }
            return dynamic_image_into_tensor(dynamic_img, info, output_buffer);
        }

        rgb8_image_buffer_to_tensor(self, info, output_buffer)
    }

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }
}

#[inline(always)]
fn dynamic_image_into_tensor(
    img: DynamicImage,
    info: &ImageToTensorInfo,
    output_buffer: &mut impl AsMut<[u8]>,
) -> Result<(), Error> {
    debug_assert!(img.width() == info.width && img.height() == info.height);
    match info.color_space {
        ImageColorSpaceType::GRAYSCALE => {
            unimplemented!()
        }
        // treat unknown as rgb8
        ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
            rgb8_image_buffer_to_tensor(&img.into_rgb8(), info, output_buffer)
        }
    }
}

#[inline(always)]
pub(super) fn rgb8_image_buffer_to_tensor<'t, Container>(
    img: &'t ImageBuffer<Rgb<u8>, Container>,
    info: &ImageToTensorInfo,
    output_buffer: &mut impl AsMut<[u8]>,
) -> Result<(), Error>
where
    Container: std::ops::Deref<Target = [u8]>,
{
    debug_assert!(
        img.width() == info.width
            && img.height() == info.height
            && info.color_space != ImageColorSpaceType::GRAYSCALE
    );

    info.normalization_options.0.get(0).unwrap_or(&1f32);

    let data_layout = info.image_data_layout;
    let res = output_buffer.as_mut();
    let mut res_index = 0;
    match info.tensor_type {
        TensorType::F32 => {
            let (r_mean, r_std, g_mean, g_std, b_mean, b_std) = get_rgb_mean_std_from_info!(info);
            let bytes = img.as_bytes();
            debug_assert_eq!(res.len(), bytes.len() * std::mem::size_of::<f32>());

            let hw = (img.width() * img.height()) as usize;
            return match data_layout {
                ImageDataLayout::NHWC => {
                    let mut i = 0;
                    while i < bytes.len() {
                        let f = ((bytes[i] as f32) - r_mean) / r_std;
                        res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                        res_index += 4;
                        let f = ((bytes[i + 1] as f32) - g_mean) / g_std;
                        res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                        res_index += 4;
                        let f = ((bytes[i + 2] as f32) - b_mean) / b_std;
                        res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                        res_index += 4;
                        i += 3;
                    }
                    Ok(())
                }
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => {
                    for start in 0..3 {
                        let mut i = start as usize;
                        while i < hw {
                            let f = ((bytes[i] as f32) - r_mean) / r_std;
                            res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                            res_index += 4;
                            i += 3;
                        }
                    }
                    Ok(())
                }
            };
        }
        TensorType::U8 => {
            let bytes = img.as_bytes();
            debug_assert_eq!(res.len(), bytes.len());
            return match data_layout {
                ImageDataLayout::NHWC => {
                    // just copy
                    res.copy_from_slice(bytes);
                    Ok(())
                }
                // batch is always 1 now
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => {
                    let hw = (img.width() * img.height()) as usize;
                    for c in 0..3 {
                        let mut i = c as usize;
                        while i < hw {
                            res[res_index] = bytes[i];
                            res_index += 1;
                            i += c;
                        }
                    }
                    Ok(())
                }
            };
        }
        _ => unimplemented!(),
    }
}
