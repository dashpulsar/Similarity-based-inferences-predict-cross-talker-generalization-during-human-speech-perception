import json

NB_PATH = 'test_nygaard_visualization.ipynb'
CEIL_CODE_PATH = 'jaeger_ceiling_cell.py'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Read the ceiling code from the separate Python file
with open(CEIL_CODE_PATH, 'r', encoding='utf-8') as f:
    ceiling_code = f.read()

# Add a trailing newline to every line and put it in a list to format it like a standard Jupyter notebook cell
code_source = [line + '\n' for line in ceiling_code.split('\n')]
if code_source and code_source[-1] == '\n':
    code_source.pop() # remove last empty newline if present

# ── New Markdown Cell (replaces cell index 4) ──
md_source = [
    "### 2. Behavioral Noise Ceiling (Jaeger Self-Predictability Method)\n",
    "\n",
    "Following Florian Jaeger's specification, the ceiling represents the **self-predictability** of the behavioral data:\n",
    "\n",
    "> *\"It is the self predictability of the data. So `glmm(cbind(n_incorrect, n_correct) ~ logodds_correct` for current item during training. Fit to training and then used during test to derive predictions, just like for the models we are comparing.\"*\n",
    "\n",
    "**Procedure (per CV fold):**\n",
    "1. **Compute per-item log-odds from training data only**: For each unique item (word × talker) in the training fold, compute proportion correct across all training subjects, then convert to log-odds: $\\text{logodds} = \\log\\frac{p}{1-p}$.\n",
    "2. **Map training-derived log-odds to test data**: Items in the test fold receive the log-odds value computed from the training fold. Scale using training-fold mean and SD (Gelman, 2008).\n",
    "3. **Fit GLMM on test data**: `glmer(cbind(numCorrect, numIncorrect) ~ logodds_scaled + (1 + logodds_scaled | SubjectID), family=binomial)` — paralleling the model comparison pipeline.\n",
    "4. **Extract Wald $z$-statistic**: Measures how strongly training-derived item difficulty predicts test-fold accuracy.\n",
    "\n",
    "**Key design principles (addressing issues in the previous implementation):**\n",
    "- **No cross-fold leakage**: predictor computed from training data only (not the entire dataset)\n",
    "- **Log-odds scale**: $\\log(p/(1-p))$, not raw accuracy, as the predictor — natural parameter space for logistic GLMM\n",
    "- **Consistent evaluation**: same GLMM fitting + cross-validation pipeline as HuBERT/MFCC/STRF comparisons"
]

new_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": md_source
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_source
}

# Replace cells 4 (markdown) and 5 (code)
nb['cells'][4] = new_md_cell
nb['cells'][5] = new_code_cell

# Remove old display cells (indices 6 and 7, in reverse order)
del nb['cells'][7]  # was loso_ceiling_vals display
del nb['cells'][6]  # was ceil_sem_pc display

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated test_nygaard_visualization.ipynb!")
print(f"  - Cell 4: Updated markdown (Jaeger method explanation)")
print(f"  - Cell 5: Replaced with Jaeger ceiling computation")
print(f"  - Cells 6-7: Removed (old display cells)")
print(f"  - Total cells: {len(nb['cells'])}")
