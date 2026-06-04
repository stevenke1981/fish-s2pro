use std::io::Cursor;
use std::sync::{Arc, Mutex};

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};

pub struct AudioPlayer {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    sink: Arc<Mutex<Option<Sink>>>,
}

impl AudioPlayer {
    pub fn new() -> Option<Self> {
        let (stream, handle) = OutputStream::try_default().ok()?;
        Some(Self {
            _stream: stream,
            handle,
            sink: Arc::new(Mutex::new(None)),
        })
    }

    pub fn play_wav_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        self.stop();
        let cursor = Cursor::new(bytes.to_vec());
        let source = Decoder::new(cursor).map_err(|e| e.to_string())?;
        let sink = Sink::try_new(&self.handle).map_err(|e| e.to_string())?;
        sink.append(source);
        sink.play();
        *self.sink.lock().unwrap() = Some(sink);
        Ok(())
    }

    pub fn stop(&self) {
        if let Some(sink) = self.sink.lock().unwrap().take() {
            sink.stop();
        }
    }
}
