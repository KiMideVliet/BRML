

import numpy as np
import pandas as pd
import math
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.causal as csl
import pyAgrum.causal.notebook as cslnb


# In[3]:


df = pd.read_csv("ai4i2020.csv")
df.info()


# In[4]:


# Renaming columns
df_renamed = df.rename(columns={'UDI': 'uid', 
                                'Product ID': 'product_id',
                                'Type': 'prod_quality',
                                'Air temperature [K]': 'air_temp',
                                'Process temperature [K]': 'process_temp',
                                'Rotational speed [rpm]': 'rot_speed',
                                'Torque [Nm]': 'torque',
                                'Tool wear [min]': 'tool_wear',
                                'Machine failure': 'machine_failure',
                                'TWF': 'tool_wear_failure',
                                'HDF': 'heat_dissipation_failure',
                                'PWF': 'power_failure',
                                'OSF': 'overstrain_failure',
                                'RNF': 'random_failure'})


# In[5]:


# Power status
def calculate_power(row):
    rad_sec = (row['rot_speed'] * 2 * math.pi) / 60
    power = row['torque'] * rad_sec
    
    if power < 3500:
        power_supply_status = 'too low'
    elif power > 9000:
        power_supply_status = 'too high'
    else:
        power_supply_status = 'normal'
    return power_supply_status
    
#df_renamed['power_supply_status'] = df_renamed.apply(calculate_power, axis=1)


# In[6]:


# Heat dissipation status
def heat_dissipation(row):
    heat_diff = abs(row['air_temp'] - row['process_temp'])
    
    if heat_diff < 8.6 and row['rot_speed'] < 1380:
        heat_diss_status = 'too high'
    else:
        heat_diss_status = 'normal'
    return heat_diss_status
    
#df_renamed['heat_diss_status'] = df_renamed.apply(heat_dissipation, axis=1)


# In[7]:


# Drop columns
df_renamed.drop(['uid', 'product_id'],axis=1,inplace=True)


# In[8]:


# Rename columns
df_renamed['tool_wear']=pd.cut(df_renamed['tool_wear'], bins = 1, precision = 0)
df_renamed['air_temp']=pd.cut(df_renamed['air_temp'], bins = 1, precision = 0)
df_renamed['rot_speed']=pd.cut(df_renamed['rot_speed'], bins = 1, precision = 0)
df_renamed['torque']=pd.cut(df_renamed['torque'], bins = 1, precision = 0)
df_renamed['process_temp']=pd.cut(df_renamed['process_temp'], bins = 1, precision = 0)


# In[9]:


# Transform column types
cat_list = ['prod_quality','air_temp','process_temp','rot_speed','torque']
bool_list = ['machine_failure','tool_wear_failure','heat_dissipation_failure','power_failure','overstrain_failure','random_failure']

df_renamed[cat_list] = df_renamed[cat_list].astype('category')
df_renamed[bool_list] = df_renamed[bool_list].astype('bool')
df_renamed.info()


# In[10]:


edge_list = 'prod_quality->tool_wear;' \
            'tool_wear->tool_wear_failure;'\
            'tool_wear_failure->machine_failure;'\
            'rot_speed->heat_dissipation_failure;'\
            'process_temp->heat_dissipation_failure;'\
            'air_temp->heat_dissipation_failure;'\
            'heat_dissipation_failure->machine_failure;'\
            'rot_speed->power_failure;'\
            'torque->power_failure;'\
            'power_failure->machine_failure;'\
            'prod_quality->overstrain_failure;'\
            'tool_wear->overstrain_failure;'\
            'torque->overstrain_failure;'\
            'overstrain_failure->machine_failure;'\
            'random_failure->machine_failure'

bn_show_relations = gum.fastBN(edge_list)

gnb.showBN(bn_show_relations)


# In[11]:


# bayesian network
bn_pred_failure = gum.BayesNet("Predict_failures")


# In[12]:


node_names_in_bn = set()
for arc in edge_list.split(';'):
    if '->' in arc:
        source, target = arc.split('->')
        node_names_in_bn.add(source.strip())
        node_names_in_bn.add(target.strip())

df_for_learning = df_renamed[list(node_names_in_bn)].copy()


# In[13]:


for col_name in df_for_learning.columns:
    if df_for_learning[col_name].dtype.name == 'category' or df_for_learning[col_name].dtype.name == 'object':
        # For categorical (including binned-turned-string-turned-category)
        labels = sorted([str(label) for label in df_for_learning[col_name].unique()])
        # Using gum.LabelizedVariable for variables with string labels
        var = gum.LabelizedVariable(col_name, col_name, labels)
        var.setDescription(f"States: {', '.join(labels)}") # Optional: add description
    elif df_for_learning[col_name].dtype.name == 'bool':
        # For boolean variables, pyAgrum can often infer states as [0, 1] or [False, True]
        # Using LabelizedVariable for consistency and explicit label definition
        var = gum.LabelizedVariable(col_name, col_name, [str(False), str(True)])
        var.setDescription("Boolean variable (False, True)") # Optional
    else:
        raise ValueError(f"Unsupported data type for column {col_name}: {df_for_learning[col_name].dtype}")
    bn_pred_failure.add(var)

# Add arcs based on the bn_string
for arc in edge_list.split(';'):
    if '->' in arc:
        source, target = arc.split('->')
        bn_pred_failure.addArc(source.strip(), target.strip())


# In[ ]:


# --- Learning Parameters ---
print("Starting parameter learning...")
# It's crucial that df_for_learning column names and types are consistent with BN variables
learner = gum.BNLearner(df_for_learning, bn_pred_failure)
learner.learn_parameters(bn_pred_failure.dag())
print("Parameter learning completed.")

# --- Inspecting the Network (Optional) ---
print("\nCPT for 'machine_failure':")
print(bn_pred_failure.cpt('machine_failure'))

print("\nCPT for 'tool_wear_failure':")
print(bn_pred_failure.cpt('tool_wear_failure'))

#If in a Jupyter notebook, you can visualize the network with learned CPTs (optional)
print("\nShowing BN structure and CPTs (if in Jupyter):")
try:
     gnb.showBN(bn_pred_failure, size="9")
     gnb.showInformation(bn_pred_failure, size="12") # Shows CPTs as well
except Exception as e:
     print(f"Could not display BN with gnb: {e}. Ensure you are in a Jupyter environment and have graphviz installed.")

# To save the learned network:
gum.saveBN(bn_pred_failure, "learned_predictive_failure_model.bifxml")
print("\nTo save the model, uncomment the gum.saveBN line.")

 # To save the structure as a DOT file for external visualization:
# You need graphviz installed on your system to convert .dot to an image (e.g., dot -Tpng model.dot -o model.png)
model_filename_dot = "learned_predictive_failure_model.dot"
gum.generateDot(bn_pred_failure, model_filename_dot) # Use gum.generateDot
print(f"Network structure saved to {model_filename_dot}. Use Graphviz to visualize (e.g., dot -Tpng {model_filename_dot} -o model.png)")
