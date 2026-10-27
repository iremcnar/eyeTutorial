# Eye → Space 👁️

A generative ASCII art animation in Python that brings a single eye to life: it opens, blinks, and then the camera dives into the pupil, flying through a twinkling starfield and into a glowing heart-shaped nebula, all drawn with text characters and no images.

<img width="1290" height="711" alt="image" src="https://github.com/user-attachments/assets/f7a08b8e-c94a-46cc-8699-33f302e3e90e" />


## Features
- **Procedural ASCII Rendering:** Builds the skin, iris, eyebrow hair, stars and nebula from NumPy noise and math instead of image files, then converts every frame into text by mapping brightness to a dark-to-bright character ramp.
- **Timeline-Driven Animation:** Smoothly eases through distinct phases: eyelid opening, blinking, wandering gaze, zooming into the pupil, and a nebula that morphs from a round cloud into a heart.
- **Layered Eye Model:** Combines eyelids, amber iris fibers, sclera veins, catchlights, eyelid crease and individually stamped eyelashes, with a pupil that acts as a portal to space.
- **Space FX & Bloom:** Features 3,200 twinkling stars flying toward the camera, shooting stars with fading tails, a slow comet, sparkle rays on bright stars, and a soft glow around bright areas.
- **Interactive GUI & Smart Redrawing:** Interactive Tkinter window with keyboard controls (`R`, `Esc`) that only redraws the cells that changed, automatically adapting to your computer's speed.
- **Included Glossary:** `Glossary.txt` explains every constant, function, variable and abbreviation used in `eye.py`, so you can read it side by side with the code.

## How to Run

```bash
# Install the only dependency (Tkinter comes bundled with Python)
pip install numpy

# Run the animation
python eye.py
```
