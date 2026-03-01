import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Always build paths relative to the project root
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "insurance.csv")
    out_dir = os.path.join(project_root, "assets", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # ---------- Figure 1: Charges distribution ----------
    plt.figure()
    df["charges"].plot(kind="hist", bins=30)
    plt.title("Distribution of Medical Insurance Charges")
    plt.xlabel("Charges")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "charges_distribution.png"), dpi=200)
    plt.close()

    # ---------- Figure 2: Charges by smoker ----------
    plt.figure()
    df.boxplot(column="charges", by="smoker")
    plt.title("Charges by Smoking Status")
    plt.suptitle("")  # removes default matplotlib subtitle
    plt.xlabel("Smoker")
    plt.ylabel("Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "charges_by_smoker.png"), dpi=200)
    plt.close()

    # ---------- Figure 3: BMI vs Charges ----------
    plt.figure()
    plt.scatter(df["bmi"], df["charges"], alpha=0.5)
    plt.title("BMI vs Charges")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bmi_vs_charges.png"), dpi=200)
    plt.close()

    print("✅ Figures saved to:", out_dir)


if __name__ == "__main__":
    main()
    print("✅ Figures saved to: assets/figures")
