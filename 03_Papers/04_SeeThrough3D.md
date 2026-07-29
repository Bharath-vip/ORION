# SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation

**Problem:** Text-to-image models struggle with occlusion reasoning, frequently generating scenes with geometrically inconsistent spatial relationships between overlapping or hidden objects.
**Why it matters:** Accurate occlusion and depth representation are crucial for realistic scene synthesis and reliable 3D layout control in generative AI.
**Method:** Introduces OSCR (Occlusion-Aware 3D Scene Representation), modeling objects as translucent 3D boxes. This explicitly encodes spatial depth, front/back relationships, and camera viewpoints to guide the generative process.
**Dataset:** Trained and evaluated on standard 3D layout-to-image generation datasets.
**Results:** Greatly enhances geometric consistency and produces realistic occlusions compared to baselines. It also allows for dynamic control over camera viewpoints while maintaining scene logic.
**Weaknesses:** Using translucent 3D boxes as proxies might not capture the intricacies of highly complex or non-cuboid object geometries, leading to imperfect masking for organic shapes.
