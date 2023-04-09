/// A rectangle with rotation in normalized coordinates. The values of box center
/// location and size are within [0, 1].
#[derive(Debug)]
pub struct NormalizedRect {
    /// Location of the center of the rectangle in image coordinates.
    /// The (0.0, 0.0) point is at the (top, left) corner.
    pub x_center: f32,
    pub y_center: f32,

    /// Size of the rectangle.
    pub height: f32,
    pub width: f32,

    /// Rotation angle is clockwise in radians. [default = 0.0]
    pub rotation: Option<f32>,

    /// Optional unique id to help associate different NormalizedRects to each other.
    pub rect_id: Option<u64>,
}
