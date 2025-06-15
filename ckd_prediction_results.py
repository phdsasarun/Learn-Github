# ===== ผลลัพธ์การใช้งาน Model ทำนายเหตุการณ์ CKD =====
# ส่วนนี้แสดงผลลัพธ์การทำนายและการประเมินประสิทธิภาพ

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# สร้างข้อมูลตัวอย่างผลลัพธ์ (ในการใช้งานจริงจะได้จากโมเดลที่ฝึกแล้ว)
np.random.seed(42)

# สร้างข้อมูลตัวอย่าง 200 ผู้ป่วย
n_samples = 200

# สร้างข้อมูลผู้ป่วยตัวอย่าง
sample_data = {
    'Patient_ID': [f'P{i:03d}' for i in range(1, n_samples + 1)],
    'Age': np.random.normal(55, 15, n_samples).astype(int),
    'BP': np.random.normal(130, 20, n_samples).astype(int),
    'Albumin': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
    'Blood_Glucose': np.random.normal(120, 40, n_samples).astype(int),
    'Serum_Creatinine': np.random.normal(1.2, 0.5, n_samples),
    'Hemoglobin': np.random.normal(12, 2, n_samples),
    'Duration_DM': np.random.exponential(5, n_samples).astype(int)
}

# แก้ไขข้อมูลให้อยู่ในช่วงที่เหมาะสม
sample_data['Age'] = np.clip(sample_data['Age'], 20, 90)
sample_data['BP'] = np.clip(sample_data['BP'], 90, 200)
sample_data['Blood_Glucose'] = np.clip(sample_data['Blood_Glucose'], 70, 300)
sample_data['Serum_Creatinine'] = np.clip(sample_data['Serum_Creatinine'], 0.5, 5.0)
sample_data['Hemoglobin'] = np.clip(sample_data['Hemoglobin'], 6, 18)
sample_data['Duration_DM'] = np.clip(sample_data['Duration_DM'], 0, 30)

df_sample = pd.DataFrame(sample_data)

# สร้างผลการทำนายตัวอย่าง
def generate_predictions(row):
    """สร้างผลการทำนายตามเกณฑ์ทางคลินิก"""
    risk_score = 0
    
    # คำนวณคะแนนความเสี่ยง
    if row['Age'] > 65: risk_score += 2
    elif row['Age'] > 45: risk_score += 1
    
    if row['BP'] > 140: risk_score += 2
    elif row['BP'] > 120: risk_score += 1
    
    if row['Serum_Creatinine'] > 1.5: risk_score += 2
    elif row['Serum_Creatinine'] > 1.2: risk_score += 1
    
    if row['Albumin'] > 2: risk_score += 2
    elif row['Albumin'] > 0: risk_score += 1
    
    if row['Hemoglobin'] < 10: risk_score += 2
    elif row['Hemoglobin'] < 12: risk_score += 1
    
    if row['Duration_DM'] > 10: risk_score += 1
    
    # แปลงเป็น 5 ระดับ
    if risk_score <= 1: return 1
    elif risk_score <= 3: return 2
    elif risk_score <= 5: return 3
    elif risk_score <= 7: return 4
    else: return 5

# สร้าง True Labels
df_sample['True_Label'] = df_sample.apply(generate_predictions, axis=1)

# สร้างผลการทำนายของทั้งสองโมเดล (เพิ่มสัญญาณรบกวนเล็กน้อย)
np.random.seed(123)

# Logistic Regression Predictions
lr_noise = np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.8, 0.1])
df_sample['LR_Prediction'] = np.clip(df_sample['True_Label'] + lr_noise, 1, 5)

# Neural Network Predictions (ดีกว่าเล็กน้อย)
nn_noise = np.random.choice([-1, 0, 1], n_samples, p=[0.05, 0.9, 0.05])
df_sample['NN_Prediction'] = np.clip(df_sample['True_Label'] + nn_noise, 1, 5)

# สร้าง Confidence Scores
df_sample['LR_Confidence'] = np.random.uniform(0.6, 0.95, n_samples)
df_sample['NN_Confidence'] = np.random.uniform(0.65, 0.98, n_samples)

# ===== 1. ภาพรวมผลการทำนาย =====
print("🎯 ===== ผลลัพธ์การใช้งาน Model ทำนายเหตุการณ์ CKD =====")
print(f"📊 จำนวนผู้ป่วยทั้งหมด: {len(df_sample)} คน")
print(f"📅 ช่วงอายุ: {df_sample['Age'].min()}-{df_sample['Age'].max()} ปี")
print(f"⏱️ ระยะเวลาเป็นเบาหวาน: {df_sample['Duration_DM'].min()}-{df_sample['Duration_DM'].max()} ปี")

# ===== 2. การแจกแจงผลการทำนาย =====
risk_levels = {
    1: "No Disease",
    2: "Low Risk", 
    3: "Moderate Risk",
    4: "High Risk",
    5: "Severe Disease"
}

print("\n📈 ===== การแจกแจงผลการทำนาย =====")

# สร้างตาราง Prediction Summary
prediction_summary = pd.DataFrame({
    'Risk_Level': list(risk_levels.keys()),
    'Risk_Description': list(risk_levels.values()),
    'True_Count': [sum(df_sample['True_Label'] == i) for i in range(1, 6)],
    'LR_Predicted': [sum(df_sample['LR_Prediction'] == i) for i in range(1, 6)],
    'NN_Predicted': [sum(df_sample['NN_Prediction'] == i) for i in range(1, 6)]
})

prediction_summary['True_Percentage'] = (prediction_summary['True_Count'] / len(df_sample) * 100).round(1)
prediction_summary['LR_Percentage'] = (prediction_summary['LR_Predicted'] / len(df_sample) * 100).round(1)
prediction_summary['NN_Percentage'] = (prediction_summary['NN_Predicted'] / len(df_sample) * 100).round(1)

print(prediction_summary.to_string(index=False))

# ===== 3. ประสิทธิภาพของโมเดล =====
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr_accuracy = accuracy_score(df_sample['True_Label'], df_sample['LR_Prediction'])
nn_accuracy = accuracy_score(df_sample['True_Label'], df_sample['NN_Prediction'])

lr_precision = precision_score(df_sample['True_Label'], df_sample['LR_Prediction'], average='weighted')
nn_precision = precision_score(df_sample['True_Label'], df_sample['NN_Prediction'], average='weighted')

lr_recall = recall_score(df_sample['True_Label'], df_sample['LR_Prediction'], average='weighted')
nn_recall = recall_score(df_sample['True_Label'], df_sample['NN_Prediction'], average='weighted')

lr_f1 = f1_score(df_sample['True_Label'], df_sample['LR_Prediction'], average='weighted')
nn_f1 = f1_score(df_sample['True_Label'], df_sample['NN_Prediction'], average='weighted')

print("\n🎯 ===== ประสิทธิภาพของโมเดล =====")
performance_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic_Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1],
    'Neural_Network': [nn_accuracy, nn_precision, nn_recall, nn_f1]
})

performance_df['Logistic_Regression'] = performance_df['Logistic_Regression'].round(4)
performance_df['Neural_Network'] = performance_df['Neural_Network'].round(4)
performance_df['Difference'] = (performance_df['Neural_Network'] - performance_df['Logistic_Regression']).round(4)

print(performance_df.to_string(index=False))

# ===== 4. สร้างกราฟแสดงผล =====
print("\n📊 ===== การสร้างกราฟแสดงผล =====")

# กราฟที่ 1: การเปรียบเทียบการแจกแจง
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('การแจกแจงจริง', 'Logistic Regression', 'Neural Network',
                   'Confusion Matrix - LR', 'Confusion Matrix - NN', 'Model Performance'),
    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
           [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "bar"}]]
)

# Bar charts สำหรับการแจกแจง
categories = list(risk_levels.values())
true_counts = prediction_summary['True_Count'].values
lr_counts = prediction_summary['LR_Predicted'].values  
nn_counts = prediction_summary['NN_Predicted'].values

fig.add_trace(go.Bar(x=categories, y=true_counts, name="True", marker_color='lightblue'), row=1, col=1)
fig.add_trace(go.Bar(x=categories, y=lr_counts, name="LR", marker_color='orange'), row=1, col=2)
fig.add_trace(go.Bar(x=categories, y=nn_counts, name="NN", marker_color='lightgreen'), row=1, col=3)

# Confusion Matrices
cm_lr = confusion_matrix(df_sample['True_Label'], df_sample['LR_Prediction'])
cm_nn = confusion_matrix(df_sample['True_Label'], df_sample['NN_Prediction'])

fig.add_trace(go.Heatmap(z=cm_lr, colorscale='Blues', showscale=False), row=2, col=1)
fig.add_trace(go.Heatmap(z=cm_nn, colorscale='Greens', showscale=False), row=2, col=2)

# Performance comparison
metrics = performance_df['Metric'].values
lr_values = performance_df['Logistic_Regression'].values
nn_values = performance_df['Neural_Network'].values

fig.add_trace(go.Bar(x=metrics, y=lr_values, name="LR", marker_color='orange'), row=2, col=3)
fig.add_trace(go.Bar(x=metrics, y=nn_values, name="NN", marker_color='lightgreen'), row=2, col=3)

fig.update_layout(height=800, title_text="CKD Prediction Results Dashboard", showlegend=False)
fig.show()

# ===== 5. การวิเคราะห์เชิงลึก =====
print("\n🔍 ===== การวิเคราะห์เชิงลึก =====")

# หาผู้ป่วยที่ทำนายผิด
lr_errors = df_sample[df_sample['True_Label'] != df_sample['LR_Prediction']]
nn_errors = df_sample[df_sample['True_Label'] != df_sample['NN_Prediction']]

print(f"❌ ผู้ป่วยที่ Logistic Regression ทำนายผิด: {len(lr_errors)} คน ({len(lr_errors)/len(df_sample)*100:.1f}%)")
print(f"❌ ผู้ป่วยที่ Neural Network ทำนายผิด: {len(nn_errors)} คน ({len(nn_errors)/len(df_sample)*100:.1f}%)")

# วิเคราะห์การทำนายผิดตาม Risk Level
print("\n📋 การทำนายผิดแยกตาม Risk Level:")
error_analysis = pd.DataFrame({
    'Risk_Level': list(risk_levels.values()),
    'LR_Errors': [len(lr_errors[lr_errors['True_Label'] == i]) for i in range(1, 6)],
    'NN_Errors': [len(nn_errors[nn_errors['True_Label'] == i]) for i in range(1, 6)]
})
print(error_analysis.to_string(index=False))

# ===== 6. ตัวอย่างการทำนายรายบุคคล =====
print("\n👤 ===== ตัวอย่างการทำนายรายบุคคล (10 คนแรก) =====")
individual_results = df_sample[['Patient_ID', 'Age', 'BP', 'Serum_Creatinine', 'Hemoglobin', 
                               'True_Label', 'LR_Prediction', 'NN_Prediction', 
                               'LR_Confidence', 'NN_Confidence']].head(10)

# เพิ่มคำอธิบายผล
individual_results['True_Risk'] = individual_results['True_Label'].map(risk_levels)
individual_results['LR_Risk'] = individual_results['LR_Prediction'].map(risk_levels)
individual_results['NN_Risk'] = individual_results['NN_Prediction'].map(risk_levels)
individual_results['LR_Confidence'] = individual_results['LR_Confidence'].round(3)
individual_results['NN_Confidence'] = individual_results['NN_Confidence'].round(3)

print(individual_results[['Patient_ID', 'Age', 'True_Risk', 'LR_Risk', 'NN_Risk', 
                         'LR_Confidence', 'NN_Confidence']].to_string(index=False))

# ===== 7. การวิเคราะห์ High-Risk Cases =====
print("\n⚠️ ===== การวิเคราะห์ผู้ป่วยกลุ่มเสี่ยงสูง =====")
high_risk_cases = df_sample[df_sample['True_Label'] >= 4]  # High Risk + Severe Disease

print(f"🔴 จำนวนผู้ป่วยเสี่ยงสูง (Level 4-5): {len(high_risk_cases)} คน")
print(f"📊 สัดส่วน: {len(high_risk_cases)/len(df_sample)*100:.1f}% ของผู้ป่วยทั้งหมด")

# ความแม่นยำในการตรวจจับผู้ป่วยเสี่ยงสูง
lr_high_risk_accuracy = len(high_risk_cases[high_risk_cases['LR_Prediction'] >= 4]) / len(high_risk_cases)
nn_high_risk_accuracy = len(high_risk_cases[high_risk_cases['NN_Prediction'] >= 4]) / len(high_risk_cases)

print(f"✅ Logistic Regression ตรวจจับผู้ป่วยเสี่ยงสูงได้: {lr_high_risk_accuracy*100:.1f}%")
print(f"✅ Neural Network ตรวจจับผู้ป่วยเสี่ยงสูงได้: {nn_high_risk_accuracy*100:.1f}%")

# ===== 8. คุณลักษณะของผู้ป่วยในแต่ละกลุ่ม =====
print("\n📈 ===== คุณลักษณะเฉลี่ยของผู้ป่วยในแต่ละระดับความเสี่ยง =====")
risk_characteristics = df_sample.groupby('True_Label').agg({
    'Age': 'mean',
    'BP': 'mean', 
    'Serum_Creatinine': 'mean',
    'Hemoglobin': 'mean',
    'Duration_DM': 'mean'
}).round(2)

risk_characteristics.index = [risk_levels[i] for i in risk_characteristics.index]
print(risk_characteristics.to_string())

# ===== 9. Confidence Score Analysis =====
print("\n🎯 ===== การวิเคราะห์ Confidence Score =====")
confidence_stats = pd.DataFrame({
    'Model': ['Logistic Regression', 'Neural Network'],
    'Mean_Confidence': [df_sample['LR_Confidence'].mean(), df_sample['NN_Confidence'].mean()],
    'Min_Confidence': [df_sample['LR_Confidence'].min(), df_sample['NN_Confidence'].min()],
    'Max_Confidence': [df_sample['LR_Confidence'].max(), df_sample['NN_Confidence'].max()],
    'Std_Confidence': [df_sample['LR_Confidence'].std(), df_sample['NN_Confidence'].std()]
}).round(3)

print(confidence_stats.to_string(index=False))

# ===== 10. การแนะนำเชิงคลินิก =====
print("\n🏥 ===== การแนะนำเชิงคลินิก =====")

# ผู้ป่วยที่ต้องติดตามเร่งด่วน
urgent_cases = df_sample[(df_sample['NN_Prediction'] >= 4) | 
                        (df_sample['LR_Prediction'] >= 4)]

print(f"🚨 ผู้ป่วยที่ต้องติดตามเร่งด่วน: {len(urgent_cases)} คน")

# ผู้ป่วยที่ผลการทำนายไม่ตรงกัน
conflicting_predictions = df_sample[abs(df_sample['LR_Prediction'] - df_sample['NN_Prediction']) >= 2]
print(f"⚡ ผู้ป่วยที่ผลการทำนายแตกต่างกันมาก: {len(conflicting_predictions)} คน")

# ===== 11. สรุปผลการใช้งาน =====
print("\n🎉 ===== สรุปผลการใช้งาน Model =====")
print(f"✅ Neural Network มีประสิทธิภาพดีกว่า Logistic Regression")
print(f"📊 ความแม่นยำโดยรวม: NN {nn_accuracy:.1%} vs LR {lr_accuracy:.1%}")
print(f"🎯 การตรวจจับผู้ป่วยเสี่ยงสูง: NN {nn_high_risk_accuracy:.1%} vs LR {lr_high_risk_accuracy:.1%}")
print(f"💡 แนะนำใช้ Neural Network สำหรับการตรวจคัดกรองเบื้องต้น")
print(f"🔍 ควรมีการตรวจสอบเพิ่มเติมสำหรับผู้ป่วย {len(urgent_cases)} คนที่มีความเสี่ยงสูง")

# ===== 12. บันทึกผลลัพธ์ =====
print("\n💾 ===== การบันทึกผลลัพธ์ =====")

# บันทึกผลการทำนายทั้งหมด
df_sample.to_csv('ckd_detailed_predictions.csv', index=False)
prediction_summary.to_csv('ckd_prediction_summary.csv', index=False)
performance_df.to_csv('ckd_model_performance.csv', index=False)
urgent_cases[['Patient_ID', 'Age', 'BP', 'Serum_Creatinine', 'True_Label', 
             'LR_Prediction', 'NN_Prediction']].to_csv('ckd_urgent_cases.csv', index=False)

print("📁 ไฟล์ที่บันทึก:")
print("   - ckd_detailed_predictions.csv: ผลการทำนายรายบุคคลทั้งหมด")
print("   - ckd_prediction_summary.csv: สรุปการแจกแจงผลการทำนาย")
print("   - ckd_model_performance.csv: ประสิทธิภาพของโมเดล")
print("   - ckd_urgent_cases.csv: รายชื่อผู้ป่วยที่ต้องติดตามเร่งด่วน")

print("\n🌟 ===== เสร็จสิ้นการแสดงผลลัพธ์ =====")

# ===== 13. Interactive Prediction Dashboard =====
print("\n🖥️ ===== Dashboard แบบ Interactive =====")

# สร้าง Dashboard ด้วย Plotly
fig_dashboard = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Model Performance Comparison', 'Risk Level Distribution',
                   'Age vs Risk Level', 'Confidence Score Distribution',  
                   'Clinical Parameters by Risk', 'Prediction Accuracy by Risk Level'),
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "box"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "bar"}]]
)

# 1. Model Performance
fig_dashboard.add_trace(
    go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
           y=[lr_accuracy, lr_precision, lr_recall, lr_f1],
           name='Logistic Regression', marker_color='orange'),
    row=1, col=1
)
fig_dashboard.add_trace(
    go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
           y=[nn_accuracy, nn_precision, nn_recall, nn_f1],
           name='Neural Network', marker_color='lightgreen'),
    row=1, col=1
)

# 2. Risk Distribution Pie Chart
fig_dashboard.add_trace(
    go.Pie(labels=list(risk_levels.values()),
           values=prediction_summary['True_Count'],
           name="Risk Distribution"),
    row=1, col=2
)

# 3. Age vs Risk Level Box Plot
for risk_level in range(1, 6):
    risk_data = df_sample[df_sample['True_Label'] == risk_level]
    fig_dashboard.add_trace(
        go.Box(y=risk_data['Age'], name=f'Level {risk_level}'),
        row=2, col=1
    )

# 4. Confidence Score Distribution
fig_dashboard.add_trace(
    go.Histogram(x=df_sample['LR_Confidence'], name='LR Confidence', 
                opacity=0.7, marker_color='orange'),
    row=2, col=2
)
fig_dashboard.add_trace(
    go.Histogram(x=df_sample['NN_Confidence'], name='NN Confidence',
                opacity=0.7, marker_color='lightgreen'),
    row=2, col=2
)

# 5. Clinical Parameters
fig_dashboard.add_trace(
    go.Scatter(x=df_sample['BP'], y=df_sample['Serum_Creatinine'],
              mode='markers',
              marker=dict(color=df_sample['True_Label'], 
                         colorscale='Viridis', size=8),
              name='BP vs Creatinine'),
    row=3, col=1
)

# 6. Accuracy by Risk Level
risk_accuracies_lr = []
risk_accuracies_nn = []
for level in range(1, 6):
    level_data = df_sample[df_sample['True_Label'] == level]
    if len(level_data) > 0:
        lr_acc = len(level_data[level_data['LR_Prediction'] == level]) / len(level_data)
        nn_acc = len(level_data[level_data['NN_Prediction'] == level]) / len(level_data)
        risk_accuracies_lr.append(lr_acc)
        risk_accuracies_nn.append(nn_acc)
    else:
        risk_accuracies_lr.append(0)
        risk_accuracies_nn.append(0)

fig_dashboard.add_trace(
    go.Bar(x=list(risk_levels.values()), y=risk_accuracies_lr,
           name='LR Accuracy', marker_color='orange'),
    row=3, col=2
)
fig_dashboard.add_trace(
    go.Bar(x=list(risk_levels.values()), y=risk_accuracies_nn,
           name='NN Accuracy', marker_color='lightgreen'),
    row=3, col=2
)

fig_dashboard.update_layout(height=1200, title_text="CKD Prediction Analysis Dashboard")
fig_dashboard.show()

print("✨ Dashboard แสดงผลเรียบร้อย!")