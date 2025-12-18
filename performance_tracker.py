import pandas as pd

df = pd.read_csv("feedback_log.csv")
validated = df[df["human_decision"].notna()]

if validated.empty:
    print("No validated feedback yet.")
else:
    acc = (validated["model_decision"]==validated["human_decision"]).mean()*100
    print("Real-world Accuracy: %.2f%%"%acc)

    print("\nAccuracy by Language:")
    print(validated.groupby("language").apply(lambda x:(x["model_decision"]==x["human_decision"]).mean()*100))

    print("\nConfusion Matrix:")
    print(pd.crosstab(validated["human_decision"], validated["model_decision"]))
