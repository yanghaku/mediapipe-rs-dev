use crate::TensorType;

use super::super::*;

/// Data layout in memory for image tensor. ```NCHW```, ```NHWC```, ```CHWN```.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum DataLayout {
    NCHW,
    NHWC,
    CHWN,
}

/// Image Color Type
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ImageColorSpaceType {
    RGB,
    GRAYSCALE,
    UNKNOWN,
}

/// Necessary information for the image to tensor.
#[derive(Debug)]
pub struct ImageToTensorInfo {
    pub color_space: ImageColorSpaceType,
    pub tensor_type: TensorType,
    pub width: u32,
    pub height: u32,
    pub stats_min: Vec<f32>,
    pub stats_max: Vec<f32>,
    /// (mean,std), len can be 1 or 3
    pub normalization_options: (Vec<f32>, Vec<f32>),
}

mod image_crate_type_impl {
    use super::*;
    use image::{DynamicImage, EncodableLayout, RgbImage};

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

    impl ToTensor<'static> for DynamicImage {
        type OutputType = Vec<u8>;

        #[inline]
        fn to_tensor(
            &self,
            input_index: usize,
            model_resource: &Box<dyn ModelResourceTrait>,
        ) -> Result<Self::OutputType, Error> {
            let info = model_resource_check_and_get_impl!(
                model_resource,
                image_to_tensor_info,
                input_index
            );
            // need resize
            if info.width != self.width() || info.height != self.height() {
                dynamic_image_into_tensor(
                    self.resize_exact(
                        info.width,
                        info.height,
                        image::imageops::FilterType::Triangle,
                    ),
                    input_index,
                    model_resource,
                    info,
                )
            } else {
                match info.color_space {
                    ImageColorSpaceType::GRAYSCALE => {
                        unimplemented!()
                    }
                    // we treat unknown as rgb8
                    ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                        if let Some(rgb) = self.as_rgb8() {
                            rgb8_image_to_tensor(rgb, input_index, model_resource, info)
                        } else {
                            rgb8_image_to_tensor(&self.to_rgb8(), input_index, model_resource, info)
                        }
                    }
                }
            }
        }
    }

    impl ToTensor<'static> for RgbImage {
        type OutputType = Vec<u8>;
        #[inline]
        fn to_tensor(
            &self,
            input_index: usize,
            model_resource: &Box<dyn ModelResourceTrait>,
        ) -> Result<Self::OutputType, Error> {
            let info = model_resource_check_and_get_impl!(
                model_resource,
                image_to_tensor_info,
                input_index
            );
            if info.width != self.width()
                || info.height != self.height()
                || info.color_space == ImageColorSpaceType::GRAYSCALE
            {
                // must resize or change to gray
                let mut dynamic_img = DynamicImage::from(self.clone());
                if info.width != self.width() || info.height != self.height() {
                    dynamic_img = dynamic_img.resize_exact(
                        info.width,
                        info.height,
                        image::imageops::FilterType::Triangle,
                    );
                }
                return dynamic_image_into_tensor(dynamic_img, input_index, model_resource, info);
            }

            rgb8_image_to_tensor(self, input_index, model_resource, info)
        }
    }

    #[inline(always)]
    fn dynamic_image_into_tensor(
        img: DynamicImage,
        input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
        info: &ImageToTensorInfo,
    ) -> Result<Vec<u8>, Error> {
        debug_assert!(img.width() == info.width && img.height() == info.height);
        match info.color_space {
            ImageColorSpaceType::GRAYSCALE => {
                unimplemented!()
            }
            // we treat unknown as rgb8
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                rgb8_image_to_tensor(&img.into_rgb8(), input_index, model_resource, info)
            }
        }
    }

    #[inline(always)]
    fn rgb8_image_to_tensor<'t>(
        img: &'t RgbImage,
        input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
        info: &ImageToTensorInfo,
    ) -> Result<Vec<u8>, Error> {
        debug_assert!(
            img.width() == info.width
                && img.height() == info.height
                && info.color_space != ImageColorSpaceType::GRAYSCALE
        );

        info.normalization_options.0.get(0).unwrap_or(&1f32);

        let tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, input_index);
        let data_layout = model_resource.data_layout();
        match tensor_type {
            TensorType::F32 => {
                let (r_mean, r_std, g_mean, g_std, b_mean, b_std) =
                    get_rgb_mean_std_from_info!(info);
                let bytes = img.as_bytes();
                let mut res = Vec::with_capacity(bytes.len() * std::mem::size_of::<f32>());
                let hw = (img.width() * img.height()) as usize;
                return match data_layout {
                    DataLayout::NHWC => {
                        let mut i = 0;
                        while i < bytes.len() {
                            let f = ((bytes[i] as f32) - r_mean) / r_std;
                            res.extend(f.to_ne_bytes());
                            let f = ((bytes[i + 1] as f32) - g_mean) / g_std;
                            res.extend(f.to_ne_bytes());
                            let f = ((bytes[i + 2] as f32) - b_mean) / b_std;
                            res.extend(f.to_ne_bytes());
                            i += 3;
                        }
                        Ok(res)
                    }
                    DataLayout::NCHW | DataLayout::CHWN => {
                        let mut i = 0 as usize;
                        while i < hw {
                            let f = ((bytes[i] as f32) - r_mean) / r_std;
                            res.extend(f.to_ne_bytes());
                            i += 3;
                        }
                        i = 1 as usize;
                        while i < hw {
                            let f = ((bytes[i] as f32) - g_mean) / g_std;
                            res.extend(f.to_ne_bytes());
                            i += 3;
                        }
                        i = 2 as usize;
                        while i < hw {
                            let f = ((bytes[i] as f32) - b_mean) / b_std;
                            res.extend(f.to_ne_bytes());
                            i += 3;
                        }
                        Ok(res)
                    }
                };
            }
            TensorType::U8 => {
                return match data_layout {
                    DataLayout::NHWC => {
                        // just copy
                        Ok(img.as_bytes().to_vec())
                    }
                    // batch is always 1 now
                    DataLayout::NCHW | DataLayout::CHWN => {
                        let bytes = img.as_bytes();
                        let mut res = Vec::with_capacity(bytes.len());
                        let hw = (img.width() * img.height()) as usize;
                        for c in 0..3 {
                            let mut i = c as usize;
                            while i < hw {
                                res.push(bytes[i]);
                                i += c;
                            }
                        }
                        Ok(res)
                    }
                };
            }
            _ => unimplemented!(),
        }
    }
}
