Any other analysis notebooks ideas that could be good implementing now? Think deeply                                                      
                                                                                                 
● Here's what I think is missing, ranked by scientific value:                                                                                         
                                                                                                                                                    
  ---                                                                                                                                                 
  High priority — directly answer the core research questions
                                                                                                                                                      
  1. Accuracy vs Moran's I scatter (accuracy_vs_morans_i.py)                                                                                        
  Join model runs with their diagnostics run via parent_run_id. Scatter plot: x=Moran's I, y=test accuracy, coloured by ρ. This directly tests whether
   topographic structure correlates with predictive performance — the central question of the project. Without this, the two axes (regularisation →
  structure, regularisation → accuracy) are only seen separately.
  2. Diversity–consistency–accuracy triangle (ensemble_tradeoffs.py)
  Join ensemble, consistency, and diversity runs on cs_hash. Plot pairs: diversity vs accuracy, consistency vs accuracy, diversity vs consistency —
  all coloured by ρ. Addresses whether diverse ensembles are less consistent, and which property is the stronger predictor of accuracy.
  3. Per-class accuracy vs ρ (per_class_accuracy_vs_rho.py)
  Load inference/{split}_inference_results.parquet for each model run (via load_inference_artifacts), compute per-class accuracy, aggregate by ρ.
  Shows which categories benefit from topographic regularisation — gives interpretable signal about what the loss actually learns.

  ---
  Medium priority — structural/qualitative insight

  4. Category profile heatmaps vs ρ (category_profiles_vs_rho.py)
  Load profiles/test_cosine_profiles.pt artifacts for a few selected ρ values and display the category×category similarity matrix side by side. Shows
  whether higher ρ produces more structured (block-diagonal, hierarchical) category representations.
  5. RSA matrix heatmap (consistency_rsa_heatmap.py)
  Load consistency/rsa_matrix.pt for a chosen ensemble and display which model pairs are most/least consistent. A deep-dive complement to the
  aggregate consistency_vs_rho.py.

  ---
  Lower priority — already partially covered

  6. Training loss curves by ρ — finegrained_rho.py partially covers this.
  7. Ensemble size vs accuracy — possible with samples3 vs samples9, but compare_groups_accuracy.py already gets close.

  ---
  The one I'd implement first is #1 — it's the simplest (one join, one scatter) and answers the most important question. #2 comes next because it uses
   the runs already computed and frames the core ensemble learning tradeoff. #3 requires loading per-run artifacts but gives the most interpretable
  result.

  Want to start with one of these?