#![feature(stdsimd)]

mod renderer;
mod vec;
mod world;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use ytesrev::prelude::*;
use ytesrev::window::{WSETTINGS_MAIN, WindowSettings};

fn main() {
    let mut wmng = WindowManager::init_window(
        renderer::Renderer::new(),
        WindowManagerSettings {
            windows: vec![
                ("renderer".into(),
                WindowSettings {
                    window_size: (renderer::SIZE as u32, renderer::SIZE as u32),
                    ..WSETTINGS_MAIN
                }
            )],
            event_step_rule: Box::new(|event| match event {
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    ..
                } => true,
                Event::MouseButtonDown { .. } => true,
                _ => false,
            }),
            quit_rule: Box::new(|event| match event {
                Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => true,
                Event::Quit { .. } => true,
                _ => false,
            }),
        },
    );

    unsafe {
        renderer::MOUSE = Some(wmng.context.mouse());
    }


    wmng.start();
}
