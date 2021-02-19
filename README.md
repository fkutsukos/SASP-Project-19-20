# Room geometry inference from acoustic echoes

In this project, we are estimating the shape of a room:​

 * The rooms are limited to shoe box shaped rooms​

 * Simulate room impulse responses (RIR) using a RIR generator based on the image source model​

 * Use multiple microphones inside the feasible region​

 * A single loudspeaker is the sound source​

​

**Idea**: Locate the set of first-order echoes from the recorded RIRs, and this set uniquely specifies the room walls with probability 1. In other words, almost surely exactly one assignment of first-order echoes to walls describes a convex room.​

The success of this task largely depends on the accurate modelling of the early reflections, since it relies on learning from which wall a particular echo originates. ​

Finally, we reconstruct the geometry of the room using the intersection of the identified walls.
