# analyze_optuna.py
import optuna
from pathlib import Path

STUDY_NAME = "optuna_solid_fe_on_1"
STORAGE = "sqlite:///outputs/optuna_solid_fe_on_1/optuna.db"
OUTDIR = Path("outputs/optuna_solid_fe_on_1/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

    if len(study.trials) == 0:
        print("Aún no hay trials registrados en la base de datos.")
        return

    print("Best trial:")
    print(f"  Value : {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")
    print(f"  Number: {study.best_trial.number}")

    # Gráficos (Plotly)
    from optuna.visualization import plot_optimization_history, plot_param_importances

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    # Mostrar (en Jupyter funciona directo; en terminal puede no abrir ventana)
    try:
        fig1.show()
        fig2.show()
    except Exception:
        pass

    # Guardar como HTML
    fig1.write_html(str(OUTDIR / "optimization_history.html"))
    fig2.write_html(str(OUTDIR / "param_importances.html"))
    print(f"Gráficos guardados en:\n  {OUTDIR/'optimization_history.html'}\n  {OUTDIR/'param_importances.html'}")

if __name__ == "__main__":
    main()

