use ndarray::Array1;

const EPSILON: f64 = 5e-2;

pub fn norm(x: &Array1<f64>) -> f64 {
    x.map(|n| n * n).sum().sqrt()
}


pub trait World: Send + Sync {
    fn distance_estimator(&self, x: &Array1<f64>) -> f64;
}


pub struct Sphere {
    pub pos: Array1<f64>,
    pub size: f64
}


impl World for Sphere {
    fn distance_estimator(&self, x: &Array1<f64>) -> f64 {
        norm(&(x - &self.pos)) - self.size
    }
}

pub struct Cube {
    pub center: Array1<f64>,
    pub dims: Array1<f64>
}


impl World for Cube {
    fn distance_estimator(&self, x: &Array1<f64>) -> f64 {
        let delta = x - &self.center;
        let delta = delta.map(|x| x.abs());

        (delta - &self.dims).iter().fold(0., |a: f64, &b| a.max(b))
    }
}


pub fn raymarch(world: &dyn World, mut x: Array1<f64>, dir: Array1<f64>) -> f64 {
    let n = norm(&dir);
    let dir = dir / n;

    for i in 0..30 {
        let de = world.distance_estimator(&x);
        if de <= EPSILON {
            let next = world.distance_estimator(&(x + de * dir));

            if next < de {
                return 1. - next / de;
            } else {
                return 0.;
            }
        }

        if de > 30. {
            return 0.;
        }

        x.scaled_add(de, &dir);
    }
    return 0.;
}
