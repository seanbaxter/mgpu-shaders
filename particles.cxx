#include "bindings.hxx"

using namespace mgpu;

// TODO: Only put simulation params here.

// simulation parameters
struct sim_params_t {
  vec3  colliderPos;
  float colliderRadius;

  vec3  gravity;
  float globalDamping;
  float particleRadius;

  uint3 gridSize;
  uint  numCells;
  vec3  worldOrigin;
  vec3  cellSize;

  uint  numBodies;
  uint  maxParticlesPerCell;

  float spring;
  float damping;
  float shear;
  float attraction;
  float boundaryDamping;
};

inline vec3 collide_spheres(vec3 posA, vec3 posB, vec3 velA, vec3 velB,
  float radiusA, float radiusB, const params_t& params) {

  vec3 rel_pos = pos_b - pos_a;
  float dist = length(rel_pos);
  float collideDist = radius_a + radius_b;

  vec3 force { };
  if(dist < collid_dist) {
    vec3 norm = rel_pos / dist;

    // relative velocity.
    vec3 relVel = velB - velA;

    // relative tangential velocity.
    vec3 tan_vel = relVel - dot(relVel, relVel) * norm;

    // spring force.
    force = -params.spring * (collideDist - dist);
    
    // dashpot (damping) fgorce
    force += params.damping * relVel;

    // tangential shear force
    force += params.shear * tanVel;

    // attraction
    force += params.attraction * relPos;
  }

  return force;
}

inline int3 calcGridPos(vec3 pos, const params_t& params) {
  int3 grid_pos = (int3)floor((p - params.worldOrigin) / params.cellSize);
  grid_pos &= params.gridSize - 1;
  return grid_pos;
}

struct system_t {
  gl_buffer_t<vec4[]> positions;
  gl_buffer_t<vec4[]> velocities;

  gl_buffer_t<vec4[]> positions_out;
  gl_buffer_t<vec4[]> velocities_out;

  // Hash each particle to a cell ID.
  gl_buffer_t<uint> cell_hash;

  // After sorting the particles, these are the offsets into the particle
  // array for each cell.
  gl_buffer_t<uint> cell_offsets;

  params_t params;
  int num_particles;

  void collide();
  void integrate();
  void sort_particles();
};

void system_t::collide() {
  auto pos_in = positions.bind_ssbo<0>;
  auto vel_out = velocities.bind_ssbo<1>;

  auto pos_out = positions.bind_ssbo<2>;
  auto vel_out = velocities.bind_ssbo<3>;

  gl_transform([=, params](int index) {
    vec3 pos = pos_data[index].xyz;
    vec3 vel = vel_data[index].xyz;

    int3 gridPos = calcGridPos(pos, sim_params_ubo);

    




  }, num_particles);
}

// Park the simulation parameters at ubo 1 and keep it there throughout the
// frame. UBO 0 is reserved for gl_transform.
[[using spirv: uniform, binding(1)]]
sim_params_t sim_params_ubo;

void system_t::integrate() {
  auto pos_data = positions.bind_ssbo<0>;
  auto vel_data = velocities.bind_ssbo<1>;

  gl_transform([=](int index) {
    sim_params_t params = sim_params_ubo;

    // Load the particle.
    vec4 pos4 = pos_data[index].xyz;
    vec4 vel4 = vel_data[index].xyz;

    vec3 pos = pos4.xyz;
    vec3 vel = vel4.xyz;

    // Apply gravity and damping.
    vel += params.gravity;
    vel *= params.global_damping;

    // Integrate the position.
    pos += vel * params.delta_time;

    // Collide with the cube sides.
    bvec3 clip_max = pos > 1 - params.particle_radius;
    pos = clip_max ? 1 - params.particle_radius : pos;
    vel *= clip_max ? params.boundary_damping : 1;

    bvec3 clip_min = pos < -1 + params.particle_radius;
    pos = clip_min ? -1 + params.particle_radius : pos;
    vel *= clip_max ? params.boundary_damping : 1;

    // Store updated terms.
    pos_data[index] = vec4(pos, pos4.w);
    vel_data[index] = vec4(vel, vel4.w);

  }, num_particles);
}

void system_t::sort_particles() {
  // Hash particles into cells.
  auto pos_data = positions.bind_ssbo<0>;
  auto hash_data = cell_hash.bind_ssbo<1>;

  gl_transform([=](int index) {
    sim_params_t params = sim_params_ubo;

    vec3 pos = pos_data[index].xyz;
    ivec3 gridPos = calcGridPos(pos, sim_params_ubo);

    uint hash = gridPos.x + params.gridSize.x * 
      (gridPos.y + params.gridSize.y * gridPos.z);

    hash_data[index] = hash;

  }, num_particles);

  // Sort particles by their hash bin. The value of the sort is the 
  // index of the parameter. This lets us gather the positions and velocities
  // after the hash sort is complete.



}