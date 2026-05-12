# Design Philosophy

Xerces is built around the idea that chess strength emerges from the interaction between search and learning, not from neural scale alone.

Rather than pursuing the largest possible networks, Xerces explores whether moderately-sized, information-dense models paired with aggressive, adaptive search can produce strong and interesting play under real compute constraints.

The project emphasizes:

- search as a first-class intelligence system
- efficient information flow through priors
- empirical refinement during search
- runtime adaptability
- systems-level optimization over benchmark chasing

Xerces treats the neural network as a powerful guide, not an oracle. The search process itself is viewed as an active computational partner capable of reshaping, amplifying, and stress-testing learned beliefs in real time.
