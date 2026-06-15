import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.interpolate import splprep, splev, splrep
from typing import Tuple

"""
Modelos de observación para captura de información.

Se definem:

- Modelo puntual.
- Modelo de cámara cónica (Conic-FOV)
- Modelo de cámara Nadir cenital

"""

@dataclass
class PathGeneratorConfig:
    base_map: np.ndarray | None = None
    number_of_waypoints_interval: tuple = (5,15)
    noise_std: float = 0.1
    max_distance_between_samples: float = 15.0
    max_distance_between_wp: float = 30.0
    min_distance_between_wp: float = 10.0

@dataclass
class ObservationConfig:
    base_map: np.ndarray | None = None
    seed: int = 0
    noise_std: float = 0.1
    path_generator_config: PathGeneratorConfig | None = None
    path_generator: object | None = None
    # For Nadir camera
    nadir_camera_radius: int = 1
    nadir_average: bool = False
    # For fov
    fov_angle: float = 60.0
    fov_radius: int = 10
    fov_min_noise_std: float = 0.05
    fov_max_noise_std: float = 0.15



class ObservationModel(ABC):

    def __init__(self, config):

        self.config = config

        self.initial_position = None  # Posición inicial del vehículo
        self.observed_mask = None  # Esto es la máscara de observación (binaria)
        self.observed_map = None # Esto es la observación del gt
        self.reset()

        np.random.seed(config.seed)

        self.path_generator = config.path_generator

        assert self.path_generator is not None, ("Se debe proporcionar un generador de trayectorias "
                                                       "para el modelo puntual.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def observation(self, ground_truth) -> np.ndarray:
        raise NotImplementedError

    def noise(self):
        raise NotImplementedError



class PointwiseObservationModel(ObservationModel):
    """
    Modelo de percepción puntual. Solo se mide en la posición en la que se está.
    """

    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        """
        Resetea la posición inicial del vehículo y el mapa de observación.
        """
        self.observed_map = np.zeros(self.config.base_map.shape)
        self.observed_mask = np.zeros(self.config.base_map.shape)
        self.initial_position = np.random.randint(0, self.config.base_map.shape[0], size=2)

    def observation(self, ground_truth) -> Tuple[np.ndarray, np.ndarray]:
        """
        Llama al generador de trayectorias y mapea las posiciones del path en observaciones.
        El generador de trayectorias garantiza puntos visitables.
        """

        assert ground_truth.shape == self.observed_map.shape, "El mapa de verdad debe tener la misma forma que el mapa de observación."

        path = self.path_generator.get_path()
        path_length = len(path)

        for t in range(path_length):

            subpath = path[:t]

            for position in subpath:

                # Lo pasamos a puntos
                x, y = position.astype(int)

                x = np.clip(x, 0, self.observed_map.shape[1]-1)
                y = np.clip(y, 0, self.observed_map.shape[0]-1)

                self.observed_mask[y, x] = 1.0
                self.observed_map[y, x] = ground_truth[y, x] + self.noise_model()

            yield self.observed_map.copy(), self.observed_mask.copy()


    def noise_model(self):
        """ Modelo de ruido - Ruido blanco gaussiano con media 0 y desviación estándar dada por el parámetro 'noise_std'."""
        noise = np.random.normal(0, self.config.noise_std)
        return noise


class NadirObservationModel(ObservationModel):

    def __init__(self, config):
        super().__init__(config)

        assert self.config.nadir_camera_radius > 0, "El radio de la cámara Nadir debe ser mayor que 0."

    def observation(self, ground_truth) -> Tuple[np.ndarray, np.ndarray]:
        """
        Llama al generador de trayectoria y calcula la observación usando un modelo de cámara de Nadir
        :param ground_truth:
        :return:
        """

        assert ground_truth.shape == self.observed_map.shape, "El mapa de verdad debe tener la misma forma que el mapa de observación."

        path = self.path_generator.get_path()
        path_length = len(path)

        for t in range(path_length):

            subpath = path[:t]

            for position in subpath:


                # Lo pasamos a puntos
                x, y = position.astype(int)

                x = np.clip(x, 0, self.observed_map.shape[1]-1)
                y = np.clip(y, 0, self.observed_map.shape[0]-1)

                # Calculamos el área de observación de la cámara Nadir
                rad = self.config.nadir_camera_radius
                y_down = np.clip(y-rad, 0, self.observed_map.shape[0]-1)
                y_up = np.clip(y+rad, 0, self.observed_map.shape[0]-1)
                x_down = np.clip(x-rad, 0, self.observed_map.shape[1]-1)
                x_up = np.clip(x+rad, 0, self.observed_map.shape[1]-1)

                crop = ground_truth[y_down:y_up+1, x_down:x_up+1]
                crop = crop + self.noise_model(crop.shape)

                if crop.size > 0:

                    self.observed_mask[y_down:y_up+1, x_down:x_up+1] = 1.0

                    if self.config.nadir_average:
                        self.observed_map[y, x] = crop.mean()
                    else:
                        self.observed_map[y_down:y_up+1, x_down:x_up+1] = crop.copy()
                else:
                    self.observed_map[y, x] = 0.0


            yield self.observed_map.copy(), self.observed_mask.copy()


    def noise_model(self, shape):

        return np.random.normal(0, self.config.noise_std, size=shape)

    def reset(self):

        """
        Resetea la posición inicial del vehículo y el mapa de observación.
        """
        self.observed_map = np.zeros(self.config.base_map.shape)
        self.observed_mask = np.zeros(self.config.base_map.shape)
        self.initial_position = np.random.randint(0, self.config.base_map.shape[0], size=2)


class ConicFOVObservationModel(ObservationModel):
    
    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        """
        Resetea la posición inicial del vehículo y el mapa de observación.
        """
        self.observed_map = np.zeros(self.config.base_map.shape)
        self.observed_mask = np.zeros(self.config.base_map.shape)
        self.initial_position = np.random.randint(0, self.config.base_map.shape[0], size=2)

    def observation(self, ground_truth) -> Tuple[np.ndarray, np.ndarray]:

        assert ground_truth.shape == self.observed_map.shape

        path = self.path_generator.get_path()
        path_length = len(path)

        for t in range(path_length):

            subpath = path[:t]

            for i, position in enumerate(subpath):

                y, x = position.astype(int)

                x = np.clip(x, 0, self.observed_map.shape[1]-1)
                y = np.clip(y, 0, self.observed_map.shape[0]-1)


                # Calcular orientación a partir del vector de movimiento
                if i < len(path) - 1:
                    dx = path[i + 1][1] - path[i][1]
                    dy = -(path[i + 1][0] - path[i][0])  # Flip Y
                    orientation = np.degrees(np.arctan2(dy, dx)) % 360
                # Último punto: mantener la orientación anterior
                else:
                    orientation = 0.0

                self._apply_fov(ground_truth, x, y, orientation)

            yield self.observed_map.copy(), self.observed_mask.copy()



        return self.observed_map.copy(), self.observed_mask.copy()

    def _apply_fov(self, ground_truth: np.ndarray, x: int, y: int, orientation: float) -> None:
        """
        Aplica el Field of View cónico centrado en (x, y) con una orientación dada.

        Args:
            ground_truth: Mapa de verdad.
            x, y:         Posición central del FoV.
            orientation:  Orientación del FoV en grados (0°=derecha, 90°=arriba).
        """
        rad = self.config.fov_radius

        # Límites del recorte en el mapa
        y_min = max(y - rad, 0)
        y_max = min(y + rad, ground_truth.shape[0] - 1)
        x_min = max(x - rad, 0)
        x_max = min(x + rad, ground_truth.shape[1] - 1)

        # Índices locales para recortar la máscara si estamos en el borde
        local_y_min = y_min - (y - rad)
        local_y_max = y_max - (y - rad)
        local_x_min = x_min - (x - rad)
        local_x_max = x_max - (x - rad)

        # Generar máscara FoV con la orientación actual
        fov_mask = self.generate_fov_mask(self.config.fov_angle, rad, orientation)


        local_mask = fov_mask[local_y_min:local_y_max + 1, local_x_min:local_x_max + 1].astype(bool)

        crop = ground_truth[y_min:y_max + 1, x_min:x_max + 1]

        if crop.size == 0 or local_mask.sum() == 0:
            self.observed_map[y, x] = 0.0
            return

        noisy_crop = crop + self.noise_model(crop.shape)

        self.observed_map[y_min:y_max + 1, x_min:x_max + 1][local_mask] = noisy_crop[local_mask]
        self.observed_mask[y_min:y_max + 1, x_min:x_max + 1][local_mask] = 1

    def noise_model(self, crop_shape) -> np.ndarray:

        rows, cols = crop_shape
        cy, cx = (rows - 1) / 2, (cols - 1) / 2

        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)  # distancia en píxeles, sin normalizar

        # Normalizar distancia a [min_std, max_std]
        min_std = self.config.fov_min_noise_std
        max_std = self.config.fov_max_noise_std
        scale = min_std + (distance / distance.max()) * (max_std - min_std)

        noise = np.random.normal(loc = 0, scale=scale)

        return noise



    @staticmethod
    def generate_fov_mask(angle: float, radius: float, orientation: float) -> np.ndarray:
        """
        Generate a binary matrix representing a conical Field of View (sector).

        Args:
            angle:       Aperture angle of the sector in degrees.
            radius:      Radius of the sector in pixels.
            orientation: Orientation of the sector's center direction in degrees.
                         0° = right, 90° = up, 180° = left, 270° = down.
        Returns:
            Binary numpy array with 1s inside the sector and 0s outside.
        """
        size = int(2 * radius) + 1
        mask = np.zeros((size, size), dtype=np.uint8)

        cy, cx = size // 2, size // 2

        # Build coordinate grids relative to center
        ys, xs = np.ogrid[:size, :size]
        dy = -(ys - cy)  # Flip Y so that 90° points up
        dx = xs - cx

        # Compute distance and angle for each pixel
        dist = np.sqrt(dx ** 2 + dy ** 2)
        pixel_angle = np.degrees(np.arctan2(dy, dx))  # [-180, 180]

        # Normalize angles to [0, 360)
        pixel_angle = pixel_angle % 360
        orientation = orientation % 360

        half_angle = angle / 2.0

        # Angular difference handling wrap-around
        diff = (pixel_angle - orientation + 180) % 360 - 180  # [-180, 180]

        # Sector condition
        in_sector = (dist <= radius) & (np.abs(diff) <= half_angle)
        mask[in_sector] = 1

        return mask






class PathGenerator(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_path(self):
        raise NotImplementedError



class RandomSplineGenerator(PathGenerator):

    def __init__(self, config : PathGeneratorConfig):
        super().__init__(config)

    def get_path(self):

        # Select a random starting point from the base map that is 1
        base_map = self.config.base_map
        h, w = base_map.shape
        start_points = np.argwhere(base_map == 1)
        start_point = start_points[np.random.choice(start_points.shape[0])]
        # Final path (row, col)
        final_path = [start_point]
        path_length = 0.0

        low_number = self.config.number_of_waypoints_interval[0]
        high_number = self.config.number_of_waypoints_interval[1]
        number_of_waypoints = np.random.randint(low_number, high_number + 1)

        while len(final_path) < number_of_waypoints:

            first_point = final_path[-1]

            # Choose all the points that are between min_distance and max_distance from the first point
            all_points = np.argwhere(base_map == 1)
            distances = np.linalg.norm(all_points - first_point, axis=1)
            valid_points = all_points[(distances >= self.config.min_distance_between_wp) &
                                      (distances <= self.config.max_distance_between_wp)]

            # Select a random point from the valid set
            second_point = valid_points[np.random.choice(valid_points.shape[0])]

            final_path.append(second_point)

            # Distancia euclídea real entre waypoints consecutivos
            path_length += float(np.linalg.norm(second_point.astype(float) - first_point.astype(float)))

        final_path = np.array(final_path)

        # Eliminar puntos duplicados consecutivos
        _, unique_idx = np.unique(final_path, axis=0, return_index=True)
        final_path_clean = final_path[np.sort(unique_idx)]

        # Interpolación paramétrica con k adaptativo
        n_points = len(final_path_clean)

        if n_points >= 2:
            k = min(3, n_points - 1)  # k=3 cúbica, k=2 cuadrática, k=1 lineal
            tck, _ = splprep([final_path_clean[:, 0], final_path_clean[:, 1]],s=1,k=k)
            num_samples = max(2, int(path_length / self.config.max_distance_between_samples))
            u_fine = np.linspace(0, 1, num=num_samples)
            x_fine, y_fine = splev(u_fine, tck)
        else:
            # Fallback: devolver el único punto repetido (caso degenerado)
            x_fine = final_path_clean[:, 0]
            y_fine = final_path_clean[:, 1]

        final_path_fine = np.asarray([x_fine, y_fine]).T

        return final_path_fine





if __name__ == "__main__":

    import matplotlib.pyplot as plt

    np.random.seed(0)

    size = 100
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    xx, yy = np.meshgrid(x, y)


    def gaussian_bump(cx, cy, sx, sy, amplitude):
        return amplitude * np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))


    bumps = [
        gaussian_bump(0.0, 0.0, 1.0, 1.0, 3.0),
        gaussian_bump(-3.0, 2.0, 0.8, 1.2, 2.0),
        gaussian_bump(3.0, -2.0, 1.5, 0.6, 1.5),
        gaussian_bump(-2.0, -3.0, 0.5, 0.5, 2.5),
        gaussian_bump(2.5, 3.0, 1.2, 0.9, 1.0),
    ]

    field = sum(bumps)

    """ Test the path generator """
    base_map = np.ones(field.shape)
    ground_truth = field

    # Create a path generator config and a path generator
    path_generator_config = PathGeneratorConfig()
    path_generator_config.base_map = base_map
    path_generator = RandomSplineGenerator(path_generator_config)
    path_generator_config.max_distance_between_samples = 20

    # Create an observation model config and an observation model
    observation_model_config = ObservationConfig()
    observation_model_config.base_map = base_map
    observation_model_config.path_generator_config = path_generator_config
    observation_model_config.path_generator = path_generator

    # Now generate the Observation map
    # observation_model = ConicFOVObservationModel(observation_model_config)
    observation_model = NadirObservationModel(observation_model_config)
    # observation_model = PointwiseObservationModel(observation_model_config)

    for observation_map, observation_mask in observation_model.observation(ground_truth):

        plt.subplot(1, 2, 1)
        plt.imshow(observation_map, vmin=0, vmax=1)

        plt.subplot(1, 2, 2)
        plt.imshow(observation_mask, vmin=0, vmax=1)
        plt.show()















































