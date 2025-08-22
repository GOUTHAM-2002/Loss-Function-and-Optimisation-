 ## Loss Functions and Optimization (Beginner-Friendly)

 A practical, story-driven intro to loss functions and optimization in AI. Start with the `NOTES.md` for a clear overview, then open the notebook to see visuals and run experiments.

 ### What’s inside
 - `NOTES.md`: concise lecture-style notes (intuition → formulas → choices)
 - `Loss_Functions_Optimization.ipynb`: hands-on visuals and simulations

 ### Quick start
 1) Create a Python environment (3.9+ recommended).
 2) Install dependencies:
    ```bash
    pip install numpy matplotlib seaborn jupyter
    ```
 3) Launch Jupyter and open the notebook:
    ```bash
    jupyter notebook
    ```
 4) Run cells from top to bottom.

 ### What you’ll learn
 - Why we need loss functions (and how they connect to likelihood)
 - MSE, MAE, Cross-Entropy, Hinge, KL (+ Huber, Focal)
 - Gradient Descent, SGD, Momentum, Adam and friends
 - How learning rate, batch size, and epochs affect training
 - Visual intuition: loss surfaces and optimization paths

 ### Tips
 - If plots don’t show, ensure you ran the setup cell and `%matplotlib inline` is enabled (Jupyter usually does this automatically).
 - Tweak learning rates and batch sizes; watch how trajectories change.
 - For imbalanced data (e.g., fraud detection), try focal loss or class weighting.

 ### Suggested order
 1) Read `NOTES.md`
 2) Run the notebook figures:
    - MSE surface → intuition for convex losses
    - CE surface → decision boundaries in logistic regression
    - Optimizer paths → GD vs Momentum vs Adam
    - SGD experiments → learning rate and batch size effects
 3) Do the practice problems at the end.

 ### License
 For learning and personal projects. If you copy sections, please link back to this repository.

