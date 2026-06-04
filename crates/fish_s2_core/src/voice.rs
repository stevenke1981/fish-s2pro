use std::path::PathBuf;

use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoiceProfile {
    pub id: Uuid,
    pub name: String,
    pub reference_wav: PathBuf,
    pub reference_text: String,
    pub created_at: DateTime<Utc>,
    pub notes: String,
}

impl VoiceProfile {
    pub fn new(name: String, reference_wav: PathBuf, reference_text: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            reference_wav,
            reference_text,
            created_at: Utc::now(),
            notes: String::new(),
        }
    }
}
