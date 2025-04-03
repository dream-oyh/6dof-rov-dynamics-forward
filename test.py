def apply_hydrodynamic_forces(self, flow_vels_w) -> TensorDict:

    body_vels = self.vel_b.clone()
    body_rpy = quaternion_to_euler(self.rot)
    flow_vels_b = torch.cat(
        [
            quat_rotate_inverse(self.rot, flow_vels_w[..., :3]),
            quat_rotate_inverse(self.rot, flow_vels_w[..., 3:]),
        ],
        dim=-1,
    )
    body_vels -= flow_vels_b  # relative velocity to the current
    # Rotate the body velocities to the NED frame
    body_vels[..., [1, 2, 4, 5]] *= -1
    body_rpy[..., [1, 2]] *= -1

    # Calculate accelerations
    body_acc = self.calculate_acc(body_vels)
    # Calculate damping forces
    damping = self.calculate_damping(body_vels.squeeze(1))
    # Calculate added mass forces
    added_mass = self.calculate_added_mass(body_acc.squeeze(1))
    # Calculate Coriolis forces
    coriolis = self.calculate_corilis(body_vels.squeeze(1))
    # Calculate Buoyancy forces
    buoyancy = self.calculate_buoyancy(body_rpy.squeeze(1))

    hydro = -(added_mass + coriolis + damping)

    # Rotate the hydrodynamic forces to the ENU frame
    hydro[:, [1, 2, 4, 5]] *= -1
    buoyancy[:, [1, 2, 4, 5]] *= -1
    hydro = hydro.unsqueeze(1)
    buoyancy = buoyancy.unsqueeze(1)

    return hydro[..., 0:3] + buoyancy[..., 0:3], hydro[..., 3:6] + buoyancy[..., 3:6]


def calculate_acc(self, body_vels):
    alpha = 0.3
    acc = (body_vels - self.prev_body_vels) / self.dt
    filteredAcc = (1.0 - alpha) * self.prev_body_acc + alpha * acc
    self.prev_body_vels = body_vels.clone()
    self.prev_body_acc = filteredAcc.clone()

    return filteredAcc


def calculate_damping(self, body_vels):
    maintained_body_vels = torch.diag_embed(body_vels)
    maintained_body_vels[:, 1, 5] = body_vels[:, 5]
    maintained_body_vels[:, 2, 4] = body_vels[:, 4]
    maintained_body_vels[:, 4, 2] = body_vels[:, 2]
    maintained_body_vels[:, 5, 1] = body_vels[:, 1]


def calculate_buoyancy(self, rpy):
    buoyancy = torch.zeros(*self.shape, 6, device=self.device)
    buoyancy.squeeze_(dim=1)
    buoyancyForce = 9.8 * self.masses[:, 0, 0] * 1.01
    dis = 0.005
    buoyancy[:, 0] = buoyancyForce * torch.sin(rpy[:, 1])
    buoyancy[:, 1] = -buoyancyForce * torch.sin(rpy[:, 0]) * torch.cos(rpy[:, 1])
    buoyancy[:, 2] = -buoyancyForce * torch.cos(rpy[:, 0]) * torch.cos(rpy[:, 1])
    buoyancy[:, 3] = -dis * buoyancyForce * torch.cos(rpy[:, 1]) * torch.sin(rpy[:, 0])
    buoyancy[:, 4] = -dis * buoyancyForce * torch.sin(rpy[:, 1])

    return buoyancy
