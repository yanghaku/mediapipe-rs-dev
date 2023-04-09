use std::fmt::{Display, Formatter};

/// Landmark represents a point in 3D space with x, y, z coordinates. The
/// landmark coordinates are in meters. z represents the landmark depth, and the
/// smaller the value the closer the world landmark is to the camera.
#[derive(Debug)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Landmark visibility. Should stay unset if not supported.
    /// Float score of whether landmark is visible or occluded by other objects.
    /// Landmark considered as invisible also if it is not present on the screen
    /// (out of scene bounds). Depending on the model, visibility value is either a
    /// sigmoid or an argument of sigmoid.
    pub visibility: Option<f32>,
    /// Landmark presence. Should stay unset if not supported.
    /// Float score of whether landmark is present on the scene (located within
    /// scene bounds). Depending on the model, presence value is either a result of
    /// sigmoid or an argument of sigmoid function to get landmark presence probability.
    pub presence: Option<f32>,
    /// Landmark name. Should stay unset if not supported.
    pub name: Option<String>,
}

/// A list of Landmarks.
#[derive(Debug)]
pub struct Landmarks {
    pub landmarks: Vec<Landmark>,
}

/// A normalized version of above Landmark struct. All coordinates should be within [0, 1].
pub type NormalizedLandmark = Landmark;

/// A list of NormalizedLandmarks.
pub type NormalizedLandmarks = Landmarks;

impl Landmark {
    pub const LANDMARK_TOLERANCE: f32 = 1e-6;
}

impl Eq for Landmark {}

impl PartialEq for Landmark {
    fn eq(&self, other: &Self) -> bool {
        return (self.x - other.x).abs() < Self::LANDMARK_TOLERANCE
            && (self.y - other.y) < Self::LANDMARK_TOLERANCE
            && (self.z - other.z) < Self::LANDMARK_TOLERANCE;
    }
}

impl Display for Landmark {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "      x:       {}", self.x)?;
        writeln!(f, "      y:       {}", self.y)?;
        writeln!(f, "      z:       {}", self.z)
    }
}

impl Display for Landmarks {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Landmarks:")?;
        if self.landmarks.is_empty() {
            return writeln!(f, "  No Landmark");
        }
        for i in 0..self.landmarks.len() {
            writeln!(f, "  Landmark #{}:", i)?;
            let l = self.landmarks.get(i).unwrap();
            write!(f, "{}", l)?;
        }
        Ok(())
    }
}
