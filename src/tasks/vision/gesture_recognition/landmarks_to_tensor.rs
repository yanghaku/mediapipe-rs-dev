use super::*;

pub fn landmarks_to_tensor(
    landmarks: &Landmarks,
    tensor_buffer: &impl AsMut<[f32]>,
    img_size: (u32, u32),
    object_normalization_origin_offset: i32,
    object_normalization: bool,
) {
}
