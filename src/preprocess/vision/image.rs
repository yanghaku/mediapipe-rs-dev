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
    #[inline(always)]
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        info: &ImageToTensorInfo,
        process_options: &ImageProcessingOptions,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        match info.color_space {
            ImageColorSpaceType::GRAYSCALE => {
                unimplemented!()
            }
            // we treat unknown as rgb8
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                if let Some(rgb) = self.as_rgb8() {
                    rgb.to_tensor(info, process_options, output_buffer)
                } else {
                    self.to_rgb8()
                        .to_tensor(info, process_options, output_buffer)
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
        process_options: &ImageProcessingOptions,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        let mut tmp_rgb_img;

        let mut rgb_img = if let Some(ref roi) = process_options.region_of_interest {
            // check roi
            let weight = self.width() as f32;
            let height = self.height() as f32;
            let x = (roi.left * weight) as u32;
            let y = (roi.top * height) as u32;
            let w = ((roi.right - roi.left) * weight) as u32;
            let h = ((roi.bottom - roi.top) * height) as u32;
            tmp_rgb_img = imageops::crop_imm(self, x, y, w, h).to_image();
            match process_options.rotation_degrees {
                0 => {}
                90 => {
                    tmp_rgb_img = imageops::rotate90(&tmp_rgb_img);
                }
                180 => {
                    imageops::rotate180_in_place(&mut tmp_rgb_img);
                }
                270 => {
                    tmp_rgb_img = imageops::rotate270(&tmp_rgb_img);
                }
                _ => unreachable!(),
            }
            &tmp_rgb_img
        } else {
            match process_options.rotation_degrees {
                0 => self,
                90 => {
                    tmp_rgb_img = imageops::rotate90(self);
                    &tmp_rgb_img
                }
                180 => {
                    tmp_rgb_img = imageops::rotate180(self);
                    &tmp_rgb_img
                }
                270 => {
                    tmp_rgb_img = imageops::rotate270(self);
                    &tmp_rgb_img
                }
                _ => unreachable!(),
            }
        };

        if info.width != rgb_img.width() || info.height != rgb_img.height() {
            tmp_rgb_img = imageops::resize(rgb_img, info.width, info.height, IMAGE_RESIZE_FILTER);
            rgb_img = &tmp_rgb_img;
        }

        if info.color_space == ImageColorSpaceType::GRAYSCALE {
            // todo: gray image
            unimplemented!()
        }

        rgb8_image_buffer_to_tensor(rgb_img, info, output_buffer)
    }

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32) {
        (self.width(), self.height())
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
