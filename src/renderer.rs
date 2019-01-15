use sdl2::pixels::PixelFormatEnum;
use sdl2::render::Canvas;
use sdl2::video::Window;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use rayon::prelude::*;

use std::f32::consts::PI;

use ytesrev::prelude::*;

use crate::vec::Vec3dx16;
use crate::world::*;

pub struct Renderer {
    data: Vec<u8>,
    rotation: f32,
    world: Box<dyn World>,
}

const SIZE: usize = 300;
const THREADS: usize = 10;

impl Renderer {
    pub fn new() -> Renderer {
        let cube = Box::new(Cube {
            center: (0., 0., 0.),
            dims: (0.5, 0.5, 0.5),
        });
        let sphere = Box::new(Sphere {
            pos: (0., 0., 0.),
            size: 0.75,
        });

        let cubesphere = Box::new(Intersection {
            objects: vec![cube, sphere],
        });

        let ground = Box::new(Plane { height: 0. });

        let world = Box::new(Union {
            objects: vec![cubesphere, ground],
        });

        Renderer {
            data: vec![0; (4 * SIZE * SIZE) as usize],
            world,
            rotation: 0.,
        }
    }
}

const FOV_DEG: f32 = 45.;
const FOV_RAD: f32 = FOV_DEG / 360. * PI;

impl Scene for Renderer {
    fn update(&mut self, dt: f64) {
        self.rotation += dt as f32 * 0.5;
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

        let rect = Rect::new(0, 0, SIZE as u32, SIZE as u32);

        canvas.copy(&texture, None, rect).expect("Can't copy");
    }

    fn event(&mut self, _event: YEvent) {}

    fn action(&self) -> Action {
        Action::Continue
    }

    fn register(&mut self) {}

    fn load(&mut self) {}
}

impl Renderer {
    fn render(&self) {
        let dones = Arc::new(AtomicUsize::new(0));

        let world = Arc::new(&*self.world);

        let camera = (self.rotation.sin() * 3., 3., -self.rotation.cos() * 3.);
        // let camera = (0., 40., -40.);

        let look = (-self.rotation, -0.6);

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

                        let xrad = look.0 + xf * FOV_RAD;
                        let yrad = look.1 - yf * FOV_RAD;

                        let dir: (f32, f32, f32) = (
                            yrad.cos() * xrad.sin(),
                            yrad.sin(),
                            yrad.cos() * xrad.cos(),
                        );

                        curr_dirs.xs = curr_dirs.xs.replace(idx, dir.0);
                        curr_dirs.ys = curr_dirs.ys.replace(idx, dir.1);
                        curr_dirs.zs = curr_dirs.zs.replace(idx, dir.2);

                        idx += 1;

                        if idx == 16 {
                            idx = 0;
                            let res16 = raymarch(&**world, Vec3dx16::from_tuple(camera), curr_dirs);

                            for i in 0..16 {
                                let y_;
                                let x_;
                                if x + i < start + 15 {
                                    y_ = y - 1;
                                    x_ = x + i - 15 + end - start;
                                } else {
                                    y_ = y;
                                    x_ = x - 15 + i;
                                }

                                unsafe {
                                    *(ptr.offset(4 * (x_ + y_ * SIZE) as isize + 0)) =
                                        res16.extract(i);
                                    *(ptr.offset(4 * (x_ + y_ * SIZE) as isize + 1)) =
                                        res16.extract(i);
                                    *(ptr.offset(4 * (x_ + y_ * SIZE) as isize + 2)) =
                                        res16.extract(i);
                                    *(ptr.offset(4 * (x_ + y_ * SIZE) as isize + 3)) = 255;
                                }
                            }
                        }
                    }
                }

                dones.fetch_add(1, Ordering::SeqCst);
            }
        });

        // Wait for each thread
        while dones.load(Ordering::SeqCst) != THREADS {
            sleep(Duration::from_millis(1));
        }
    }
}
