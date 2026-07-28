# Notebooks

Run Jupyter from the repository root:

```bash
jupyter lab
```

Run the notebooks in this order:

1. `01_site_selection.ipynb`  
   Builds candidate features, scores locations, selects sites and exports
   agent-ready results.

2. `02_build_agent_tools.ipynb`  
   Loads the deterministic exports and wraps them as LlamaIndex tools.

3. `03_multiagent_workflow.ipynb`  
   Runs the specialist-agent workflow and validates the final structured
   report.

4. `04_evaluate_system.ipynb`  
   Evaluates tool behaviour, agent tool selection, workflow routing and
   evidence grounding.

The notebooks use:

```python
PROJECT_ROOT = Path.cwd().resolve()
```

Therefore, start Jupyter from the repository root rather than from inside
the `notebooks/` directory.

Raw data availability is documented in `data/DATA_GUIDE.md`; the
notebooks do not display separate file-status or preflight tables.
