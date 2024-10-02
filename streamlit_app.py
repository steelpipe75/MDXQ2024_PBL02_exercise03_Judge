import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder


def app():
    st.set_page_config(
        page_title="PBL02 Judge App",
        layout="wide"
    )
    st.title("PBL02 演習03 採点アプリ")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("正解データ")
        answer_file = st.file_uploader("9/17にCompetitionサイトで公開された正解データ(answer.tsv)をアップロード", type="tsv")

    with col2:
        st.subheader("投稿データ")
        submit_file = st.file_uploader("投稿ファイル(tsv)をアップロード", type="tsv")

    with col3:
        AGGRID_TYPE = "Streamlit AgGrid形式 で 表示"
        DEFAULT_TYPE = "Streamlit 標準形式 で 表示"
        st.subheader("詳細データの表示タイプ")
        view_type = st.radio(label="表示タイプ選択", options=[AGGRID_TYPE, DEFAULT_TYPE], label_visibility="hidden")

    st.divider()

    if answer_file is not None:
        anser_df = pd.read_csv(answer_file, header=None, sep="\t")
        anser_df.columns = ["FileName","Category","Actual(LE)"]

    if submit_file is not None:
        submit_df = pd.read_csv(submit_file, header=None, sep="\t")
        submit_df.columns = ["FileName","Predicted(LE)"]

    if answer_file is not None and submit_file is not None:
        merged_df = pd.merge(anser_df, submit_df)

        # 正誤判定
        merged_df["Result"] = "誤答"
        merged_df.loc[merged_df["Actual(LE)"] == merged_df["Predicted(LE)"], "Result"] = "正答"

        # 混同行列のどの要素に該当するかラベル付け
        merged_df["Class"] = "FP"
        merged_df.loc[(merged_df["Actual(LE)"] == merged_df["Predicted(LE)"]) & (merged_df["Actual(LE)"] == 1), "Class"] = "TP"
        merged_df.loc[(merged_df["Actual(LE)"] == merged_df["Predicted(LE)"]) & (merged_df["Actual(LE)"] == 0), "Class"] = "TN"
        merged_df.loc[(merged_df["Actual(LE)"] != merged_df["Predicted(LE)"]) & (merged_df["Actual(LE)"] == 1), "Class"] = "FN"

        # 評価データをまとめる表を用意
        result_df = pd.DataFrame({
            "TP": [0, 0, 0],
            "FP": [0, 0, 0],
            "FN": [0, 0, 0],
            "TN": [0, 0, 0],
            "Accuracy":  [0.0, 0.0, 0.0],
            "Recall":    [0.0, 0.0, 0.0],
            "Precision": [0.0, 0.0, 0.0],
            "F1-score":  [0.0, 0.0, 0.0],
        })
        result_df.index = ["public", "private", "all"]

        # 混同行列の各要素の数を表にまとめる
        for cls in ["TP", "FP", "FN", "TN"]:
            result_df.loc["all", cls] = sum(merged_df["Class"] == cls)
            for cat in ["public", "private"]:
                # st.text(f"cat = {cat}, cls = {cls}") # 動作確認用表示
                result_df.loc[cat, cls] = sum((merged_df["Category"] == cat) & (merged_df["Class"] == cls))

        # 各種評価指標のスコアを表にまとめる
        cat = "all"
        result_df.loc[cat, "Accuracy"] = accuracy_score(merged_df["Actual(LE)"], merged_df["Predicted(LE)"])
        result_df.loc[cat, "Recall"] = recall_score(merged_df["Actual(LE)"], merged_df["Predicted(LE)"])
        result_df.loc[cat, "Precision"] = precision_score(merged_df["Actual(LE)"], merged_df["Predicted(LE)"])
        result_df.loc[cat, "F1-score"] = f1_score(merged_df["Actual(LE)"], merged_df["Predicted(LE)"])

        for cat in ["public", "private"]:
            cat_df = merged_df[ merged_df["Category"] == cat ]
            result_df.loc[cat, "Accuracy"] = accuracy_score(cat_df["Actual(LE)"], cat_df["Predicted(LE)"])
            result_df.loc[cat, "Recall"] = recall_score(cat_df["Actual(LE)"], cat_df["Predicted(LE)"])
            result_df.loc[cat, "Precision"] = precision_score(cat_df["Actual(LE)"], cat_df["Predicted(LE)"])
            result_df.loc[cat, "F1-score"] = f1_score(cat_df["Actual(LE)"], cat_df["Predicted(LE)"])

        # ラベル⇒数値 (Streamlit AgGridでのフィルタが数値型しかできない？フィルタ用に対応する数値列を追加)
        le_cat = LabelEncoder()
        merged_df["Category(LE)"] = le_cat.fit_transform(merged_df["Category"])
        le_class = LabelEncoder()
        merged_df["Class(LE)"] = le_class.fit_transform(merged_df["Class"])
        le_result = LabelEncoder()
        merged_df["Result(LE)"] = le_result.fit_transform(merged_df["Result"])

        # 数値⇒ラベル (1がどっちだったか忘れることあるため変換)
        merged_df["Actual"] = "良品"
        merged_df.loc[merged_df["Actual(LE)"] == 1, "Actual"] = "不良品"
        merged_df["Predicted"] = "良品"
        merged_df.loc[merged_df["Predicted(LE)"] == 1, "Predicted"] = "不良品"

        new_order = [
            "FileName",
            "Category","Actual","Predicted","Result","Class",
            "Category(LE)","Actual(LE)","Predicted(LE)","Result(LE)","Class(LE)"
        ]
        merged_df = merged_df[new_order]

        st.subheader("スコアデータ")
        st.dataframe(result_df)

        st.subheader("詳細データ")
        if view_type == AGGRID_TYPE:
            AgGrid(merged_df, fit_columns_on_grid_load=True)
        else:
            st.dataframe(merged_df)

if __name__ == "__main__":
    app()
