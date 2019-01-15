use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::render::{BlendMode, Canvas};
use sdl2::video::Window;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use rayon::prelude::*;

use std::f32::consts::PI;

use ytesrev::prelude::*;

use crate::vec::Vec3dx16;
use crate::world::{raymarch, Sphere, World};
use packed_simd::f32x16;

pub struct Renderer {
    data: Vec<u8>,
    rotation: f32,
    world: Box<dyn World>,
}

const SIZE: usize = 300;
const THREADS: usize = 4;

const CAMERA_POSES: (f32, f32, f32) = (0., 0., -40.);

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            data: vec![0; (4 * SIZE * SIZE) as usize],
            world: Box::new(Sphere {
                pos: (0., 0., 0.),
                size: 5.,
            }),
            rotation: 0.,
        }
    }
}

const FOV_DEG: f32 = 45.;
const FOV_RAD: f32 = FOV_DEG / 360. * PI;

impl Scene for Renderer {
    fn update(&mut self, dt: f64) {
        self.rotation += dt as f32 * 0.2;
    }

    fn draw(&self, canvas: &mut Canvas<Window>, _settings: DrawSettings) {
        self.render();

        let creator = canvas.texture_creator();
        let mut texture = creator
            .create_texture_target(Some(PixelFormatEnum::ABGR8888), SIZE as u32, SIZE as u32)
            .expect("Can't make texture");

        // texture.set_blend_mode(BlendMode::Blend);

        texture
            .update(None, self.data.as_slice(), 4 * SIZE)
            .expect("Can't update");

        let (w, h) = canvas.window().size();

        let rect = Rect::new(0, 0, h, h);

        canvas.copy(&texture, None, rect).expect("Can't copy");
    }

    fn event(&mut self, event: YEvent) {}

    fn action(&self) -> Action {
        Action::Continue
    }

    fn register(&mut self) {}

    fn load(&mut self) {}
}

impl Renderer {
    fn render(&self) {
        let start = Instant::now();

        let dones = Arc::new(AtomicUsize::new(0));

        let world = Arc::new(&*self.world);

        (0..THREADS).into_par_iter().for_each({
            let ptr: *mut u8 = self.data.as_ptr() as *mut u8;
            let aptr = AtomicPtr::new(ptr);

            let dones = dones.clone();
            let world = world.clone();

            move |n| {
                let ptr = aptr.load(Ordering::SeqCst);

                let start = n * SIZE / THREADS;
                let end = (n + 1) * SIZE / THREADS;

                let mut curr_dirs = Vec3dx16::default();

                let mut idx = 0;

                for y in 0..SIZE {
                    for x in start..end {
                        let xf = x as f32 / (SIZE - 1) as f32 * 2. - 1.;
                        let yf = y as f32 / (SIZE - 1) as f32 * 2. - 1.;

                        let xrad = (xf + self.rotation.sin()) * FOV_RAD;
                        let yrad = yf * FOV_RAD;

                        let dir: (f32, f32, f32) = (
                            yrad.cos() * xrad.sin(),
                            -yrad.sin(),
                            yrad.cos() * xrad.cos(),
                        );

                        curr_dirs.xs = curr_dirs.xs.replace(idx, dir.0);
                        curr_dirs.ys = curr_dirs.ys.replace(idx, dir.1);
                        curr_dirs.zs = curr_dirs.zs.replace(idx, dir.2);

                        idx += 1;

                        if idx == 16 {
                            idx = 0;
                            let res16 =
                                raymarch(&**world, Vec3dx16::from_tuple(CAMERA_POSES), curr_dirs);

                            for i in 0..16 {
                                unsafe {
                                    *(ptr.offset(4 * (x + y * SIZE + i - 15) as isize + 0)) =
                                        res16.extract(i);
                                    *(ptr.offset(4 * (x + y * SIZE + i - 15) as isize + 1)) =
                                        res16.extract(i);
                                    *(ptr.offset(4 * (x + y * SIZE + i - 15) as isize + 2)) =
                                        res16.extract(i);
                                    *(ptr.offset(4 * (x + y * SIZE + i - 15) as isize + 3)) = 255;
                                }
                            }
                        }
                    }
                }

                dones.fetch_add(1, Ordering::SeqCst);
            }
        });

        // Wait for each thread
        while dones.load(Ordering::SeqCst) != THREADS {}

        // println!("Time: {:?}", Instant::now() - start);
    }
}
