#!/usr/bin/env python
# coding: utf-8

# <table align="center" width=100%>
#     <tr>
#         <td width="15%">
#             <img src="house.jpg">
#         </td>
#         <td>
#             <div align="center">
#                 <font color="#21618C" size=24px>
#                     <b>Property Price Prediction
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# ## Problem Statement
# 
# A key challenge for property sellers is to determine the sale price of the property. The ability to predict the exact property value is beneficial for property investors as well as for buyers to plan their finances according to the price trend. The property prices depend on the number of features like the property area, basement square footage, year built, number of bedrooms, and so on. Regression analysis can be useful in predicting the price of the house.

# ## Data Definition
# 
# **Dwell_Type:** Identifies the type of dwelling involved in the sale
# 
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
# 
# **Zone_Class:** Identifies the general zoning classification of the sale
# 		
#        A	Agriculture
#        C	Commercial
#        FV   Floating Village Residential
#        I	Industrial
#        RH   Residential High Density
#        RL   Residential Low Density
#        RP   Residential Low Density Park 
#        RM   Residential Medium Density
# 	
# **LotFrontage:** Linear feet of street-connected to the property
# 
# **LotArea:** Lot size is the lot or parcel side where it adjoins a street, boulevard or access way
# 
# **Road_Type:** Type of road access to the property
# 
#        Grvl	Gravel	
#        Pave	Paved
#        	
# **Alley:** Type of alley access to the property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	 No alley access
# 		
# **Property_Shape:** General shape of the property
# 
#        Reg	Regular	
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular
#        
# **LandContour:** Flatness of the property
# 
#        Lvl	Near Flat/Level	
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
# 		
# **Utilities:** Type of utilities available
# 		
#        AllPub	All public Utilities (E, G, W and S)	
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	   Electricity only	
# 	
# **LotConfig:** Lot configuration
# 
#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac   Cul-de-sac
#        FR2	   Frontage on 2 sides of property
#        FR3	   Frontage on 3 sides of property
# 	
# **LandSlope:** Slope of property
# 		
#        Gtl	Gentle slope
#        Mod	Moderate Slope	
#        Sev	Severe Slope
# 	
# **Neighborhood:** Physical locations within Ames city limits
# 
#        Blmngtn	Bloomington Heights
#        Blueste	Bluestem
#        BrDale	 Briardale
#        BrkSide	Brookside
#        ClearCr	Clear Creek
#        CollgCr	College Creek
#        Crawfor	Crawford
#        Edwards	Edwards
#        Gilbert	Gilbert
#        IDOTRR	 Iowa DOT and Rail Road
#        MeadowV	Meadow Village
#        Mitchel	Mitchell
#        Names	  North Ames
#        NoRidge	Northridge
#        NPkVill	Northpark Villa
#        NridgHt	Northridge Heights
#        NWAmes	 Northwest Ames
#        OldTown	Old Town
#        SWISU	  South & West of Iowa State University
#        Sawyer	 Sawyer
#        SawyerW	Sawyer West
#        Somerst	Somerset
#        StoneBr	Stone Brook
#        Timber	 Timberland
#        Veenker	Veenker
# 			
# **Condition1:** Proximity to various conditions
# 	
#        Artery  Adjacent to an arterial street
#        Feedr   Adjacent to feeder street	
#        Norm	Normal	
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to positive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 	
# **Condition2:** Proximity to various conditions (if more than one is present)
# 		
#        Artery   Adjacent to an arterial street
#        Feedr    Adjacent to feeder street	
#        Norm	 Normal	
#        RRNn     Within 200' of North-South Railroad
#        RRAn	 Adjacent to North-South Railroad
#        PosN     Near positive off-site feature--park, greenbelt, etc.
#        PosA     Adjacent to positive off-site feature
#        RRNe	 Within 200' of East-West Railroad
#        RRAe     Adjacent to East-West Railroad
# 	
# **Dwelling_Type:** Type of dwelling
# 		
#        1Fam	  Single-family Detached	
#        2FmCon	Two-family Conversion; originally built as a one-family dwelling
#        Duplx	 Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit
# 	
# **HouseStyle:** Style of dwelling
# 	
#        1Story	One story
#        1.5Fin	One and one-half story: 2nd level finished
#        1.5Unf	One and one-half story: 2nd level unfinished
#        2Story	Two-story
#        2.5Fin	Two and one-half story: 2nd level finished
#        2.5Unf	Two and one-half story: 2nd level unfinished
#        SFoyer	Split Foyer
#        SLvl	  Split Level
# 	
# **OverallQual:** Rates the overall material and finish of the house
# 
#        10   Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
# 	
# **OverallCond:** Rates the overall condition of the house
# 
#        10   Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average	
#        5	Average
#        4	Below Average	
#        3	Fair
#        2	Poor
#        1	Very Poor
# 		
# **YearBuilt:** Original construction date
# 
# **YearRemodAdd:** Remodel date (same as construction date if no remodeling or additions)
# 
# **RoofStyle:** Type of roof
# 
#        Flat	   Flat
#        Gable	  Gable
#        Gambrel	Gabrel (Barn)
#        Hip	    Hip
#        Mansard	Mansard
#        Shed	   Shed
# 		
# **RoofMatl:** Roof material
# 
#        ClyTile	Clay or Tile
#        CompShg	Standard (Composite) Shingle
#        Membran	Membrane
#        Metal	  Metal
#        Roll	   Roll
#        Tar&Grv	Gravel & Tar
#        WdShake	Wood Shakes
#        WdShngl	Wood Shingles
# 		
# **Exterior1st:** Exterior covering on the house
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	 Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	  Other
#        Plywood	Plywood
#        PreCast	PreCast	
#        Stone	  Stone
#        Stucco	 Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 	
# **Exterior2nd:** Exterior covering on the house (if more than one material)
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	 Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	  Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	  Stone
#        Stucco	 Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 	
# **MasVnrType:** Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace   Brick Face
#        CBlock	Cinder Block
#        None	  None
#        Stone     Stone
# 	
# **MasVnrArea:** Masonry veneer area in square feet
# 
# **ExterQual:** Evaluates the quality of the material on the exterior
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# **ExterCond:** Evaluates the present condition of the material on the exterior
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# **Foundation:** Type of foundation
# 		
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	 Poured Concrete	
#        Slab	  Slab
#        Stone	 Stone
#        Wood	  Wood
# 		
# **BsmtQual:** Evaluates the height of the basement
# 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 		
# **BsmtCond:** Evaluates the general condition of the basement
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 	
# **BsmtExposure:** Refers to walkout or garden level walls
# 
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Minimum Exposure
#        No	No Exposure
#        NA	No Basement
# 	
# **BsmtFinType1:** Rating of basement finished area
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinished
#        NA	 No Basement
# 		
# **BsmtFinSF1:** Type 1 finished square feet
# 
# **BsmtFinType2:** Rating of basement finished area (if multiple types)
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinished
#        NA	 No Basement
# 
# **BsmtFinSF2:** Type 2 finished square feet
# 
# **BsmtUnfSF:** Unfinished square feet of the basement area
# 
# **TotalBsmtSF:** Total square feet of the basement area
# 
# **Heating:** Type of heating
# 		
#        Floor   Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace	
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace
# 		
# **HeatingQC:** Heating quality and condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# **CentralAir:** Central air conditioning
# 
#        N	No
#        Y	Yes
# 		
# **Electrical:** Electrical system
# 
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	  Mixed
# 		
# **1stFlrSF:** First Floor square feet
#  
# **2ndFlrSF:** Second floor square feet
# 
# **LowQualFinSF:** Low quality finished square feet (all floors)
# 
# **GrLivArea:** Above grade (ground) living area square feet
# 
# **BsmtFullBath:** Basement full bathrooms
# 
# **BsmtHalfBath:** Basement half bathrooms
# 
# **FullBath:** Full bathrooms above grade
# 
# **HalfBath:** Half baths above grade
# 
# **Bedroom:** Bedrooms above grade (does NOT include basement bedrooms)
# 
# **Kitchen:** Kitchens above grade
# 
# **KitchenQual:** Kitchen quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        	
# **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms)
# 
# **Functional:** Home functionality (Assume typical unless deductions are warranted)
# 
#        Typ	 Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	 Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	 Severely Damaged
#        Sal	 Salvage only
# 		
# **Fireplaces:** Number of fireplaces
# 
# **FireplaceQu:** Fireplace quality
# 
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in the main level
#        TA	Average - Prefabricated Fireplace in the main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in a basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
# 		
# **GarageType:** Garage location
# 		
#        2Types	More than one type of garage
#        Attchd	Attached to the home
#        Basment   Basement Garage
#        BuiltIn   Built-In (Garage part of the house - typically has hte room above garage)
#        CarPort   Car Port
#        Detchd	Detached from home
#        NA	    No Garage
# 		
# **GarageYrBlt:** Year garage was built
# 		
# **GarageFinish:** Interior finish of the garage
# 
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	 No Garage
# 		
# **GarageCars:** Size of garage in car capacity
# 
# **GarageArea:** Size of garage in square feet
# 
# **GarageQual:** Garage quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# **GarageCond:** Garage condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# **PavedDrive:** Paved driveway
# 
#        Y	Paved 
#        P	Partial Pavement
#        N	Dirt/Gravel
# 		
# **WoodDeckSF:** Wood deck area in square feet
# 
# **OpenPorchSF:** Open porch area in square feet
# 
# **EnclosedPorch:** Enclosed porch area in square feet
# 
# **3SsnPorch:** Three season porch area in square feet
# 
# **ScreenPorch:** Screen porch area in square feet
# 
# **PoolArea:** Pool area in square feet
# 
# **PoolQC:** Pool quality
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
# 		
# **Fence:** Fence quality
# 		
#        GdPrv   Good Privacy
#        MnPrv   Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	  No Fence
# 	
# **MiscFeature:** Miscellaneous feature not covered in other categories
# 		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	  None
# 		
# **MiscVal:** Value of miscellaneous feature
# 
# **MoSold:** Month Sold (MM)
# 
# **YrSold:** Year Sold (YYYY)
# 
# **SaleType:** Type of sale
# 		
#        WD 	  Warranty Deed - Conventional
#        CWD	  Warranty Deed - Cash
#        VWD      Warranty Deed - VA Loan
#        New      Home just constructed and sold
#        COD	  Court Officer Deed/Estate
#        Con	  Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	  Other
# 		
# **SaleCondition:** Condition of sale
# 
#        Normal	Normal Sale
#        Abnorml   Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand   Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#        Family	Sale between family members
#        Partial   Home was not completed when last assessed (associated with New Homes)
#        
# **Property_Sale_Price:** Price of the house

# ## Icon Legends
# <table>
#   <tr>
#     <th width="25%"> <img src="infer.png" style="width:25%;"></th>
#     <th width="25%"> <img src="alsoreadicon.png" style="width:25%;"></th>
#     <th width="25%"> <img src="todo.png" style="width:25%;"></th>
#     <th width="25%"> <img src="quicktip.png" style="width:25%;"></th>
#   </tr>
#   <tr>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Inferences from outcome</b></font></div>
#     </td>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Additional Reads</b></font></div>
#     </td>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Lets do it</b></font></div>
#     </td>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Quick Tips</b></font></div>
#     </td>
# 
# </tr>
# 
# </table>

# ## Table of Contents
# 
# 1. **[Import Libraries](#import_lib)**
# 2. **[Set Options](#set_options)**
# 3. **[Read Data](#Read_Data)**
# 4. **[Prepare and Analyze the Data](#data_preparation)**
#     - 4.1 - [Understand the Data](#Data_Understanding)
#         - 4.1.1 - [Data Type](#Data_Types)
#         - 4.1.2 - [Summary Statistics](#Summary_Statistics)
#         - 4.1.3 - [Distribution of Variables](#distribution_variables)
#         - 4.1.4 - [Discover Outliers](#outlier)
#         - 4.1.5 - [Missing Values](#Missing_Values)
#         - 4.1.6 - [Correlation](#correlation)
#         - 4.1.7 - [Analyze Relationships Between Target and Categorical Variables](#cat_num)
#     - 4.2 - [Prepare the Data](#Data_Preparation)
#         - 4.2.1 - [Check for Normality](#Normality)
#         - 4.2.2 - [Dummy Encode the Categorical Variables](#dummy)
# 5. **[Linear Regression (OLS)](#LinearRegression)**
#     - 5.1 - [Multiple Linear Regression - Full Model - with Log Transformed Dependent Variable (OLS)](#withLog)
#     - 5.2 - [Multiple Linear Regression - Full Model - without Log Transformed Dependent Variable (OLS)](#withoutLog)
#     - 5.3 - [Feature Engineering](#Feature_Engineering)
#       - 5.3.1 - [Multiple Linear Regression (Using New Feature1) - Full Model (OLS)](#feature1)
#       - 5.3.2 - [Multiple Linear Regression (Using New Feature2) - Full Model (OLS)](#feature2)
# 6. **[Feature Selection](#feature_selection)**
#      - 6.1 - [Variance Inflation Factor](#vif)
#      - 6.2 - [Forward Selection](#forward)
#      - 6.3 - [Backward Elimination](#backward)
#      - 6.4 - [Recursive Feature Elimination (RFE)](#rfe)
# 7. **[Conclusion and Interpretation](#conclusion)**

# <a id='import_lib'></a>
# # 1. Import Libraries

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Import the required libraries and functions</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


get_ipython().system('pip install statsmodels --upgrade')


# In[3]:


get_ipython().system('pip3 uninstall statsmodels -y')


# In[4]:


get_ipython().system('pip3 install statsmodels==0.10.0rc2 --pre --user')


# In[5]:


get_ipython().system('pip install statsmodels')


# In[6]:


# suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# 'Os' module provides functions for interacting with the operating system 
import os

# 'Pandas' is used for data manipulation and analysis
import pandas as pd 

# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
import matplotlib.pyplot as plt

# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns

# 'Scikit-learn' (sklearn) emphasizes various regression, classification and clustering algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing

# import 'is_string_dtype' and 'is_numeric_dtype' to check the data type 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# 'Statsmodels' is used to build and analyze various statistical models
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
from statsmodels.compat import lzip
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 'SciPy' is used to perform scientific computations
from scipy.stats import shapiro
from scipy import stats

# import functions to perform feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE


# In[7]:


# set the plot size using 'rcParams'
# once the plot size is set using 'rcParams', it sets the size of all the forthcoming plots in the file
# pass width and height in inches to 'figure.figsize' 
plt.rcParams['figure.figsize'] = [15,8]


# <a id='set_options'></a>
# # 2. Set Options

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Now we make necessary changes to :<br><br>
# 1. Display complete data frames<br>
# 2. To avoid the exponential number<br>
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[8]:


# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None

# use below code to convert the 'exponential' values to float
np.set_printoptions(suppress=True)


# <a id='RD'></a>
# # 3. Read Data

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Read and display data to get an insight into the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[9]:


# read csv file using pandas
df_house = pd.read_csv('HousePrices.csv')

# display the top 5 rows of the dataframe
df_house.head()


# #### Lets take a glance at our dataframe and see how it looks

# #### Dimensions of the data

# In[10]:


# 'shape' function gives the total number of rows and columns in the data
df_house.shape


# <a id='data_preparation'></a>
# # 4. Data Analysis and Preparation

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Data preparation is the process of cleaning and transforming raw data before building predictive models. <br><br>
#                         Here, we analyze and prepare data to perform classification techniques:<br>
#                         1. Check data types. Ensure your data types are correct. Refer data definitions to validate <br>
#                         2. If data types are not as per business definition, change the data types as per requirement <br>
#                         3. Study summary statistics<br>
#                         4. Distribution of variables<br>
#                         5. Study correlation<br>
#                         6. Detect outliers<br>
#                         7. Check for missing values<br><br>
#                         Note: It is an art to explore data, and one needs more and more practice to gain expertise in this area
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Data_Understanding'></a>
# ## 4.1 Understand the Dataset

# <a id='Data_Types'></a>
# ### 4.1.1 Data Type
# The main data types in Pandas dataframes are the object, float, int64, bool, and datetime64. To understand each attribute of our data, it is always good for us to know the data type of each column.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In our dataset, we have numerical and categorical variables. The numeric variables should have data type 'int'/'float' while categorical variables should have data type 'object'.<br><br> 
#                         1. Check for the data type <br>
#                         2. For any incorrect data type, change the data type with the appropriate type<br>
#                         3. Recheck for the data type
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Check for the data type**

# In[11]:


# 'dtypes' gives the data type for each column
df_house.dtypes


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>From the above output, we see that the data type of 'Dwell_Type', 'OverallQual' and 'OverallCond' is 'int64'.<br>
# 
# But according to data definition, 'Dwell_Type ', 'OverallQual' and 'OverallCond' are categorical variables, which are wrongly interpreted as 'int64', so we will convert these variables data type to 'object'.</br></b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# #### Let us remove the Id column as this will not be necessary for our analysis

# In[11]:


df_house.drop(['Id'], axis=1, inplace=True)


# <a id='Summary_Statistics'></a>
# ### 4.1.2 Summary Statistics
# 
# Here we take a look at the summary of each attribute. This includes the count, mean, the minimum and maximum values as well as some percentiles for numeric variables and count, unique, top, frequency for categorical variables.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> In our dataset we have both numerical and categorical variables. Now we check for summary statistics of all the variables<br><br>
#                         1. For numerical variables, use the describe()<br>
#                         2. For categorical variables, use the describe(include=object)
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. For numerical variables, use the describe()**

# In[12]:


# the describe() returns the statistical summary of the variables
# by default, it returns the summary of numerical variables
df_house.describe()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
# <b>The above output illustrates the summary statistics of all the numeric variables like mean, median (50%), standard deviation, minimum, and maximum values, along with the first and third quantiles.<br><br> 
#     The LotFrontage ranges from 21 feet to 313 feet, with mean 70 feet. It can be seen that the oldest house was built in 1872 and the recent house built was in 2010. <br><br>Note that the minimum pool area is 0 sq.ft. From this we can infer that not all houses have pools and yet have been considered to calculate the mean pool area. Also the count for LotFrontage is less than the total number of observations which indicates the presence of missing values. We deal with the missing data in section 4.1.4 
#     </b>     </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **2. For categorical features, use the describe(include=object)**

# In[13]:


# summary of categorical variables
df_house.describe(include=object)

# Note: If we pass 'include=object' to the .describe(), it will return descriptive statistics for categorical variables only


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>The summary statistics for categorical variables contains information about the total number of observations, number of unique classes, the most occurring class, and its frquency.:<br><br> 
#                         We will understand the outputs of the above table using variable 'MSZoning' <br> 
#                         count: Number of observations i.e., 1460 <br> 
#                         unique: Number of unique values or classes in the variable. i.e., it has 5 classes in it.<br>  
#                         top: The most occurring class, in this variable it is RL (Residential Low Density) <br>
#                         frequency: Frequency of the most repeated class; out of 1460 observations RL has a frequency of 1151 <br> Also some of the variables have count less than total number of observations which indicates the presence of missing values. We deal with the missing data in section 4.1.4 </b>  
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# As, `PoolQC` have only 7 non-zero values out of 1460 observations. And, the variable `PoolArea` contains the area of these 7 pools; thus, we remove the variables `PoolQC` and `PoolArea`.

# In[14]:


# use drop() to drop the redundant variables
# 'axis = 1' drops the corresponding columns
df_house = df_house.drop(['PoolQC', 'PoolArea'], axis= 1)

# re-check the shape of the dataframe
df_house.shape


# <a id='distribution_variables'></a>
# ### 4.1.3 Distribution of Variables

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Check the distribution of all the variables <br><br>
#                         1. Distribution of numeric variables<br>
#                         2. Distribution of categoric variables
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Distribution of numeric variables**
# 
# we plot the histogram to check the distribution of the variables.

# In[15]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[16]:


# plot the histogram of numeric variables
# the hist() function considers the numeric variables only, by default
# rotate the x-axis labels by 20 degree using the parameter, 'xrot'
df_house.hist(xrot = 20, )

# adjust the subplots
plt.tight_layout()

# display the plot
plt.show()  


# #### Visualize the target variable

# In[17]:


# Sale Price Frequency Distribution
# set the xlabel and the fontsize
plt.xlabel("Sale Price", fontsize=15)

# set the ylabel and the fontsize
plt.ylabel("Frequency", fontsize=15)

# set the title of the plot
plt.title("Sale Price Frequency Distribution", fontsize=15)

# plot the histogram for the target variable
plt.hist(df_house["Property_Sale_Price"])
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The above plot shows that the target variable 'Property_Sale_Price' is positively skewed. 
#                     </b>   
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# **2. Distribution of categoric variables.**

# For the categoric variables, we plot the countplot to check the distribution of the each category in the variable.

# In[22]:


categorical


# In[21]:


# create a list of all categorical variables
# initiate an empty list to store the categorical variables
categorical=[]

# use for loop to check the data type of each variable
for column in df_house:

    # use 'if' statement with condition to check the categorical type 
    if is_string_dtype(df_house[column]):
        
        # append the variables with 'categoric' data type in the list 'categorical'
        categorical.append(column)

# plot the count plot for each categorical variable 
# set the number of rows in the subplot using the parameter, 'nrows'
# set the number of columns in the subplot using the parameter, 'ncols'
# 'figsize' sets the figure size
fig, ax = plt.subplots(7, 6, figsize = (50, 35))

# use for loop to plot the count plot for each variable
for variable, subplot in zip(categorical, ax.flatten()):
    
    # use countplot() to plot the graph
    # pass the axes for the plot to the parameter, 'ax'
    sns.countplot(df_house[variable], ax = subplot)
    
# display the plot
plt.show()


# #### Boxplot of OverallQuality and Property_Sale_Price

# In[23]:


# draw the boxplot for OverallQuality and the Property_Sale_Price
sns.boxplot(y="Property_Sale_Price", x="OverallQual", data= df_house)

# set the title of the plot and the fontsize
plt.title("Overall Quality vs Property_Sale_Price", fontsize=15)

# set the xlabel and the fontsize
plt.xlabel("Overall Quality", fontsize=15)

# set the ylabel and the fontsize
plt.ylabel("Sale Price", fontsize=15)

# display the plot
plt.show()


# #### Boxplot of Overall Condition and Property_Sale_Price

# In[24]:


# draw the boxplot for OverallQuality and the Property_Sale_Price
sns.boxplot(y="Property_Sale_Price", x="OverallCond", data= df_house)

# set the title of the plot and the fontsize
plt.title("Overall Condition vs Property_Sale_Price", fontsize=15)

# set the xlabel and the fontsize
plt.xlabel("Overall Condition", fontsize=15)

# set the ylabel and the fontsize
plt.ylabel("Sale Price", fontsize=15)

# display the plot
plt.show()


# #### Draw the pairplot of the numeric variables

# In[25]:


# Pairplot of numeric variables

# select the columns for the pairplot
columns= ["Property_Sale_Price", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd"]

# draw the pairplot such that the diagonal should be density plot and the other graphs should be scatter plot
sns.pairplot(df_house[columns], size=2, kind= "scatter", diag_kind="kde")

# display the plot
plt.show()


# <a id='outlier'></a>
# ### 4.1.4 Outliers Discovery

# In[26]:


# plot a boxplot of target variable to detect the outliers
sns.boxplot(df_house['Property_Sale_Price'], color='coral')

# set plot label
# set text size using 'fontsize'
plt.title('Distribution of Target Variable (Property_Sale_Price)', fontsize = 15)

# display the plot
plt.show()


# The above plot shows that there are extreme observations in the target variable. As these values can affect the prediction of the regression model, we remove such observations before building the model.

# In[27]:


# remove the observations with the house price greater than or equal to 500000
# consider all the observations with the Property_Sale_Price less than 500000
df_house = df_house[df_house['Property_Sale_Price'] < 500000]

# check the dimension of the data
df_house.shape


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The number of observations is reduced to 1451 from 1460 which suggests that we have removed the 9 observations with extremely high house price.
#                     </b>   
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# <a id='Missing_Values'></a>
# ### 4.1.5 Missing Values
# 
# If the missing values are not handled properly we may end up drawing an inaccurate inference about the data.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to get the count of missing values in each column.<br><br>
#                         <ol type="1"><li>Check the missing values</li>
#                             <li>Visualize missing values using heatmap</li>
#                             <li>Handle missing values
#                             <ul type="i">
#                                 <li>For numeric variables</li>
#                                 <li> For categoric variables</li>
#                             </ul>
#                         </ol>  </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Check the missing values**

# In[28]:


# sort the variables on the basis of total null values in the variable
# 'isnull().sum()' returns the number of missing values in each variable
# 'ascending = False' sorts values in the descending order
# the variable with highest number of missing values will appear first
Total = df_house.isnull().sum().sort_values(ascending = False)          

# calculate the percentage of missing values
# 'ascending = False' sorts values in the descending order
# the variable with highest percentage of missing values will appear first
Percent = (df_house.isnull().sum()*100/df_house.isnull().count()).sort_values(ascending = False)   

# concat the 'Total' and 'Percent' columns using 'concat' function
# 'keys' is the list of column names
# 'axis = 1' concats along the columns
missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    

# add the column containing data type of each variable
missing_data['Type'] = df_house[missing_data.index].dtypes
missing_data


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>The variables 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType', and 'Electrical' contains the missing values.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **2. Visualize missing values using heatmap**

# In[29]:


# plot heatmap to visualize the null values in each column
# 'cbar = False' does not show the color axis 
sns.heatmap(df_house.isnull(), cbar=False)

# display the plot
plt.show()


# The horizontal lines in the heatmap correspond to the missing values. This is a visual representation of the previous output.

# **3. Handle missing values**

# **Replace the missing values in numerical variables**

# In[30]:


# consider the numeric variables with missing values
# pass the condition to filter the variables with number of missing values greater than zero and numeric data type
num_missing_values = missing_data[(missing_data['Total'] > 0) & (missing_data['Type'] != 'object')]
num_missing_values


# For the numerical variables, replace the missing values by their respective mean, median or mode as per the requirement.

# In[31]:


# the variable 'LotFrontage' is positively skewed
# fill the missing values with its median value using fillna()
df_house['LotFrontage'] = df_house['LotFrontage'].fillna(df_house['LotFrontage'].median())


# The dataframe `missing_data` shows that, all the variables containing the garage information have 81 missing values. This indicates that there are 81 observations for which garage facility is not available. Thus, we replace the missing values in the numeric variable `GarageYrBlt` by 0.

# In[32]:


# replace missing values in 'GarageYrBlt' with 0 using fillna() 
df_house['GarageYrBlt'] = df_house['GarageYrBlt'].fillna(0)


# In[33]:


# the variable 'MasVnrArea' is positively skewed
# fill the missing values with its median value using fillna()
df_house['MasVnrArea'] = df_house['MasVnrArea'].fillna(df_house['MasVnrArea'].median())


# **Replace the missing values in categorical variables**

# In[34]:


# consider the categoric variables with missing values
# pass the condition to filter the variables with number of missing values greater than zero and categoric data type
cat_missing_values = missing_data[(missing_data['Total'] > 0) & (missing_data['Type'] == 'object')]
cat_missing_values


# In[35]:


# according to the data definition, 'NA' denotes the absence of miscellaneous feature
# replace NA values in 'MiscFeature' with a valid value, 'None'
df_house['MiscFeature'] = df_house['MiscFeature'].fillna('None')


# In[36]:


# according to the data definition, 'NA' denotes the absence of alley access
# replace NA values in 'Alley' with a valid value, 'No alley access' 
df_house['Alley'] = df_house['Alley'].fillna('No alley access')


# In[37]:


# according to the data definition, 'NA' denotes the absence of fence
# replace NA values in 'Fence' with a valid value, 'No Fence'
df_house['Fence'] = df_house['Fence'].fillna('No Fence')


# In[38]:


# according to the data definition, 'NA' denotes the absence of fireplace
# replace null values in 'FireplaceQu' with a valid value, 'No Fireplace' 
df_house['FireplaceQu'] = df_house['FireplaceQu'].fillna('No Fireplace')


# The dataframe `missing_data` shows that, all the variables containing the garage information have 81 missing values. This indicates that there are 81 observations for which garage facility is not available. Thus, we replace the missing values in the categoric variables representing the garage by `No Garage`.

# In[39]:


# use 'for loop' to replace NA values in the below columns with a valid value, 'No Garage' 
# 'inplace = True' replace the null values in the original data
for col in ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']:
    df_house[col].fillna('No Garage', inplace = True)


# In[40]:


# according to the data definition, 'NA' denotes the absence of basement in the variabels 'BsmtQual', 'BsmtCond', 
# 'BsmtExposure', 'BsmtFinType1','BsmtFinType2'
# use 'for loop' to replace NA values with 'No Basement' in these columns 
# 'inplace = True' replace the null values in the original data
for col in ['BsmtFinType2', 'BsmtExposure', 'BsmtQual','BsmtCond','BsmtFinType1']:
    df_house[col].fillna('No Basement', inplace = True)


# In[41]:


# according to the data definition, 'NA' denotes the absence of masonry veneer
# replace NA values in 'MasVnrType' with a valid value, 'None'
df_house['MasVnrType'] = df_house['MasVnrType'].fillna('None')


# In[42]:


# replace the null values in 'Electrical' with its mode
# calculate the mode of the 'Electrical'
mode_electrical = df_house['Electrical'].mode()

# print mode of the 'Electrical'
print(mode_electrical)


# In[43]:


# replace NA values in 'Electrical' with its mode, i.e. 'SBrkr'
df_house['Electrical'].fillna('SBrkr' , inplace = True)


# After replacing the null values for the required variables, recheck the null values. 

# In[44]:


# recheck the null values
# plot heatmap to visualize the null values in each column
# 'cbar = False' does not show the color axis 
sns.heatmap(df_house.isnull(), cbar=False)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The above plot shows that there are no missing values in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='correlation'></a>
# ### 4.1.6 Study correlation

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> To check the correlation between numerical variables, we perform the following steps:<br><br>
#                     1. Compute a correlation matrix  <br>
#                     2. Plot a heatmap for the correlation matrix
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Compute a correlation matrix**

# In[45]:


# use the corr() function to generate the correlation matrix of the numeric variables
corrmat = df_house.corr()

# print the correlation matrix
shape.corrmat


# **2. Plot the heatmap for the diagonal correlation matrix**

# A correlation matrix is a symmetric matrix. Plot only the upper triangular entries using a heatmap.

# In[46]:


correlation=df_house.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(correlation[(correlation >= 0.5) | (correlation <= -0.5)],
            annot=True,linewidths=.1,linecolor="blue",vmax = 1, vmin = -1)
plt.title('Correlation between features', fontsize=15)
plt.show()


# In[ ]:


# set the plot size
# pass the required height and width to the parameter, 'figsize'  
plt.figure(figsize = (30,20))

# use 'mask' to plot a upper triangular correlation matrix 
# 'tril_indices_from' returns the indices for the lower-triangle of matrix
# 'k = -1' consider the diagonal of the matrix
mask = np.zeros_like(corrmat)
mask[np.triu_indices_from(mask, k = 1)] = True

# plot the heat map
# corr: give the correlation matrix
# cmap: color code used for plotting
# vmax: gives a maximum range of values for the chart
# vmin: gives a minimum range of values for the chart
# annot: prints the correlation values in the chart
# annot_kws: sets the font size of the annotation
# mask: mask the upper traingular matrix values
sns.heatmap(corrmat, cmap = 'Blues', vmax = 1.0, vmin = -1.0, annot = True, annot_kws = {"size": 12}, mask = mask)

# set the size of x and y axes labels
# set text size using 'fontsize'
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>The diagonal entries are all '1' which represents the correlation of the variable with itself. The dark blue squares represent the variables with strong positive correlation. <br><br>The above plot shows that there is highest positive correlation (= 0.88) between the variables 'GarageArea' and 'GarageCars'. Also there is strong positive correlation between the pairs (1StFlrSF, TotalBsmtSF) and (TotRmsAbvGrd, GrlivArea). These variables may involved in multicollinearity.<br>
#                         No two variables have strong negative correlation in the dataset.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Correlation does not imply causation. In other words, if two variables are correlated, it does not mean that one variable caused the other.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="alsoreadicon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>I love to know more:</b> <br><br>
#                     <a href="https://bit.ly/2PBvA8T">Why correlation does not imply causation </a>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# <a id='cat_num'></a>
# ### 4.1.7 Analyze Relationships Between Target and Categorical Variables

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Plot the box-and-whisker plot for visualizing relationships between target and categorical variables.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[47]:


# create a list of all categorical variables
# initiate an empty list to store the categorical variables
categorical=[]

# use for loop to check the data type of each variable
for column in df_house:
    
    # use 'if' statement with condition to check the categorical type 
    if is_string_dtype(df_house[column]):
        
        # append the variables with 'categoric' data type in the list 'categorical'
        categorical.append(column)

# plot the boxplot for each categorical and target variable 
# set the number of rows in the subplot using the parameter, 'nrows'
# set the number of columns in the subplot using the parameter, 'ncols'
# 'figsize' sets the figure size
fig, ax = plt.subplots(nrows = 14, ncols = 3, figsize = (40, 100))

# use for loop to plot the boxplot for each categoric and target variable
for variable, subplot in zip(categorical, ax.flatten()):
    
    # use boxplot() to plot the graph
    # pass the axes for the plot to the parameter, 'ax'
    sns.boxplot(x = variable, y = 'Property_Sale_Price', data = df_house, ax = subplot)
    
# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Most of the categorical variables have impact on the selling price of the house. The median selling price is exponentially increasing with respect to the rating of the overall quality of the material used.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <a id='Data_Preparation'></a>
# ## 4.2 Prepare the Data

# <a id='Normality'></a>
# ### 4.2.1 Check for Normality

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> In order to check for normality of our target variable, <br><br>
#                         1. Plot a histogram and also perform the Shapiro-Wilk test <br>
#                         2. If the data is not normally distributed, use log transformation to get near normally distributed data <br>
#                         3. Recheck for normality by plotting histogram and performing Shapiro-Wilk
#                         test
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 
# 

# **1. Plot a histogram and also perform the Shapiro-Wilk test**

# To plot a histogram, we use the `hist()` from the matplotlib library.

# In[48]:


# check the distribution of target variable
df_house.Property_Sale_Price.hist()

# add plot and axes labels
# set text size using 'fontsize'
plt.title('Distribution of Target Variable (Property_Sale_Price)', fontsize = 15)
plt.xlabel('Sale Price', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>From the above plot, we can notice that the variable 'Property_Sale_Price' is right skewed and not normally distributed.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We should not only make conclusions through visual representations or only using a statistical test but perform multiple ways to get the best insights.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# Let us perform from Shapiro-Wilk test to check the normality of the target variable.

# The null and alternate hypothesis of Shapiro-Wilk test is as follows: <br>
# 
# <p style='text-indent:25em'> <strong> H<sub>o</sub>: The data is normally distributed</strong> </p>
# <p style='text-indent:25em'> <strong> H<sub>1</sub>: The data is not normally distributed</strong> </p>

# In[49]:


# normality test using shapiro()
# the test returns the the test statistics and the p-value of the test
stat, p = shapiro(df_house.Property_Sale_Price)

# to print the numeric outputs of the Shapiro-Wilk test upto 3 decimal places
# %.3f: returns the a floating point with 3 decimal digit accuracy
# the '%' holds the place where the number is to be printed
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

# display the conclusion
# set the level of significance to 0.05
alpha = 0.05

# if the p-value is greater than alpha print we accept alpha 
# if the p-value is less than alpha print we reject alpha
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It is apparent that the p-value is less than 0.05. So we have enough evidence to reject the null hypothesis. It can be concluded that the data is not normally distributed.<br><br>
#                         Now we opt for log transformation in order to reduce the skewness. We will log transform the claim variable. </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Shaprio Wilk Test does not work if the number of observations are more than 5000. However Shapiro Wilk test is more robust than other tests. In case where the observations are more than 5000, other tests like Anderson Darling test or Jarque Bera test may also be used.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **2. If the data is not normally distributed, use log transformation to get near normally distributed data**
# 
# As mentioned above we opt for log transformation. The log transformation can be used to make highly skewed distributions less skewed. We use `np.log()` to log transform the 'Property_Sale_Price' variable. We also store the transformed variable into our data frame with a new name, `log_Property_Sale_Price`.

# In[50]:


# log transformation for normality using np.log()
df_house['log_Property_Sale_Price'] = np.log(df_house['Property_Sale_Price'])

# display first 5 rows of the data
df_house.head()


# **3. Recheck for normality by plotting histogram and performing Shapiro-Wilk test**
# 
# Let us first plot a histogram of `log_Property_Sale_Price`.

# In[51]:


# recheck for normality 
# plot the histogram using hist
df_house.log_Property_Sale_Price.hist()

# add plot and axes labels
# set text size using 'fontsize'
plt.title('Distribution of Log-transformed Target Variable (log_Property_Sale_Price)', fontsize = 15)
plt.xlabel('Sale Price (log-transformed)', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It can be seen that the variable log_Property_Sale_Price is near normally distributed. However we again confirm by Shapiro-Wilk test.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# Let us perform Shapiro-Wilk test.

# In[52]:


# check the normality by Shapiro-Wilk test
# the test returns the the test statistics and the p-value of the test
stat, pv = shapiro(df_house['log_Property_Sale_Price'])

# to print the numeric outputs of the Shapiro-Wilk test upto 3 decimal places
# %.3f: returns the a floating point with 3 decimal digit accuracy
# the '%' holds the place where the number is to be printed
print('Statistics=%.3f, p-value=%.3f' % (stat, pv))

# display the conclusion
# set the level of significance to 0.05
alpha = 0.05

# if the p-value is greater than alpha print we accept alpha 
# if the p-value is less than alpha print we reject alpha
if pv > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# In[53]:


# find the skewness of the variable log_Property_Sale_Price
df_house['log_Property_Sale_Price'].skew()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It can be visually seen that the data has near-normal distribution, but Shapiro-Wilk test does not support the claim.
# <br>                    
# Note that in reality it might be very tough for your data to adhere to all assumptions your algorithm needs.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='dummy'></a>
# ### 4.2.2 Dummy Encode the Categorical Variables

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> We need to perform dummy encoding on our categorical variables before we proceed; since the method of OLS works only on the numeric data. <br><br>
#                     In order to dummy encode, we do the following: <br>
#                     1. Filter numerical and categorical variables<br>
#                     2. Dummy encode the catergorical variables<br>
#                     3. Concatenate numerical and dummy encoded categorical variables</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Filter numerical and categorical variables**

# In[54]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[55]:


# filter the categorical features in the dataset using select_dtypes()
# include=[np.object]: selects the categoric features
df_categoric_features = df_house.select_dtypes(include = object)

# display categorical features
df_categoric_features.columns


# **2. Dummy encode the catergorical variables**

# In[56]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **3. Concatenate numerical and dummy encoded categorical variables**

# In[57]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_house_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_house_dummy.head()


# In[58]:


# check the shape of the dataframe
df_house_dummy.shape


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Thus we have obtained the dummy coded variables. <br><br>
#                         Note: The categorical variables are dummy encoded creating n-1 variables for each categorical variables, where n is the number of classes in each categorical variable.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>There are various forms of encoding like n-1 dummy encoding, one hot encoding, label encoding, frequency encoding.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="alsoreadicon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>I love to know more:</b> <br><br>
#                     <a href="https://bit.ly/36nZQKg">1. FAQ: What is Dummy Coding? <br>
#                     <a href="https://bit.ly/2q9Omt9">2. Encoding Categorical Features
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Let us now proceed to train models. We shall begin by fitting a linear regression model using the method of ordinary least square(OLS). </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='LinearRegression'></a>
# # 5. Linear Regression (OLS)

# <a id='withLog'></a>
# ## 5.1 Multiple Linear Regression - Full Model - with Log Transformed Dependent Variable (OLS)

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to build the model, we do the following: <br><br>
#                        1. Split the data into training and test sets<br>
#                        2. Build model using sm.OLS().fit()<br>
#                        3. Predict the values using test set <br>
#                        4. Compute accuracy measures <br>
#                        5. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **1. Split the data into training and test sets**

# Statmodels linear regression function (OLS) does not include the intercept term by default. Thus, we add the intercept column to the dataset.

# In[59]:


# add the intercept column using 'add_constant()'
df_house_dummy = sm.add_constant(df_house_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_house_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_house_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **2. Build model using sm.OLS().fit()**

# In[60]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_withlog = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()


# In[61]:


# print the summary output
linreg_full_model_withlog.summary()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 94.9% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 1.955 and indicates that there is no autocorrelation. The Condition Number 3.23e+19 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> Condition Number : One way to assess multicollinearity is to compute the condition number(CN). If CN is less than 100, there is no multicollinearity. If CN is between 100 and 1000, there is moderate multicollinearity and if CN is greater 1000 there is severe multicollinearity in the data. <br><br>
#                         Durbin-Watson : The Durbin-Watson statistic will always have a value between 0 and 4. A value of 2.0 means that there is no autocorrelation detected in the sample. Values from 0 to less than 2 indicate positive autocorrelation and values from from 2 to 4 indicate negative autocorrelation.</b>     
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **3. Predict the values using test set**

# In[20]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_withlog_predictions = linreg_full_model_withlog.predict(X_test)


# In[66]:


linreg_full_model_withlog_predictions.count()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[19]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_withlog_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **4. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[18]:


# calculate rmse using rmse()
linreg_full_model_withlog_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_withlog_rsquared = linreg_full_model_withlog.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_withlog_rsquared_adj = linreg_full_model_withlog.rsquared_adj 


# **5. Tabulate the results**

# In[ ]:


# create the result table for all accuracy scores
# accuracy measures considered for model comparision are RMSE, R-squared value and Adjusted R-squared value
# create a list of column names
cols = ['Model', 'RMSE', 'R-Squared', 'Adj. R-Squared']

# create a empty dataframe of the colums
# columns: specifies the columns to be selected
result_tabulation = pd.DataFrame(columns = cols)

# compile the required information
linreg_full_model_withlog_metrics = pd.Series({'Model': "Linreg full model with log of target variable ",
                     'RMSE':linreg_full_model_withlog_rmse,
                     'R-Squared': linreg_full_model_withlog_rsquared,
                     'Adj. R-Squared': linreg_full_model_withlog_rsquared_adj     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_withlog_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Let us also take a look at building a linear regression full model without performing any kind of transformation on target variable.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='withoutLog'></a>
# ## 5.2 Multiple Linear Regression - Full Model - without Log Transformed Dependent Variable (OLS)

# In this section we build a full model with linear regression using OLS (Ordinary Least Square) technique. By full model we indicate that we consider all the independent variables that are present in the dataset.
# 
# In this case, we do not consider any kind of transformation on the dependent variable, we use the 'Property_Sale_Price' variable as it is.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> We do not need to do the train and test split again since it has been done while building the previous model<br><br>
#                        In order to build the model, we do the following: <br>
#                        1. Build model using sm.OLS().fit()<br>
#                        2. Predict the values using test set <br>
#                        3. Compute accuracy measures <br>
#                        4. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the Property_Sale_Price as the target variable
# use fit() to fit the model on train data
linreg_full_model_withoutlog = sm.OLS(y_train['Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_full_model_withoutlog.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 94.4% of the variation in dependent variable Property_Sale_Price. The Durbin-Watson test statistics is 2.002 and indicates that there is no autocorrelation. The Condition Number 3.23e+19 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **2. Predict the values using test set**

# In[ ]:


# predict the 'Property_Sale_Price' using predict()
linreg_full_model_withoutlog_predictions = linreg_full_model_withoutlog.predict(X_test)


# **3. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_withoutlog_rmse = rmse(actual_Property_Sale_Price, linreg_full_model_withoutlog_predictions)

# calculate R-squared using rsquared
linreg_full_model_withoutlog_rsquared = linreg_full_model_withoutlog.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_withoutlog_rsquared_adj = linreg_full_model_withoutlog.rsquared_adj 


# **4. Tabulate the results**

# In[ ]:


# append the result table 
# compile the required information
linreg_full_model_withoutlog_metrics = pd.Series({'Model': "Linreg full model without log of target variable ",
                                                 'RMSE':linreg_full_model_withoutlog_rmse,
                                                 'R-Squared': linreg_full_model_withoutlog_rsquared,
                                                 'Adj. R-Squared': linreg_full_model_withoutlog_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_withoutlog_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>On comparing the above models, it is seen that the R-squared and the Adjusted R-squared value for the model considering log transformation of the variable 'Property_Sale_Price' is higher than the other model. However, the RMSE value of the model without considering the log transformation is considerably higher. So, we continue with variable 'log_Property_Sale_Price', instead of 'Property_Sale_Price'.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Let us also take a look at building a linear regression full model by adding new features to the dataset. 
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Feature_Engineering'></a>
# ## 5.3 Feature Engineering

# It is the process of creating new features using domain knowledge of the data that provides more insight into the data. Let us create a few features from the existing dataset and build a regression model on the newly created data.

# <a id='feature1'></a>
# ### 5.3.1 Multiple Linear Regression (Using New Feature1) - Full Model (OLS)

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to build the model, we do the following: <br><br>
#                        1. Create a new feature by using variables 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', and 'GrLivArea' <br>
#                        2. Split the data into train and test sets<br>
#                        3. Build model using sm.OLS().fit()<br>
#                        4. Predict the values using test set <br>
#                        5. Compute accuracy measures <br>
#                        6. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Create a new feature by using variables 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', and 'GrLivArea'.**

# **Calculate the complete area of the house.**<br>
# Create a new variable `TotalSF` representing the total square feet area of the house by adding the area of the first floor, second floor, ground level and basement of the house.

# In[ ]:


# create a new variable 'TotalSF' using the variables 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', and 'GrLivArea'
# add the new variable to the dataframe 'df_house'
df_house['TotalSF'] = df_house['TotalBsmtSF'] + df_house['1stFlrSF'] + df_house['2ndFlrSF'] + df_house['GrLivArea']

# since we have added a new variable using the existing variables
# remove the existing variables in the data, 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF' and 'GrLivArea'
# use 'drop()' to remove the variables
# 'axis = 1' drops the specific columns
df_house = df_house.drop(["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea"], axis=1)


# In[ ]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# filter the categorical features in the dataset using select_dtypes()
# include=[np.object]: selects the categoric features
df_categoric_features = df_house.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_feature1 = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_full_model_feature1.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 94.9% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 1.955 and indicates that there is no autocorrelation. The Condition Number 1.26e+16 suggests that there is severe multicollinearity in the data. This condition number is higher than the condition number in the full model without feature engineering.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **4. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_feature1_predictions = linreg_full_model_feature1.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_feature1_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **5. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_feature1_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_feature1_rsquared = linreg_full_model_feature1.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_feature1_rsquared_adj = linreg_full_model_feature1.rsquared_adj 


# **6. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_full_model_feature1_metrics = pd.Series({'Model': "Linreg with new feature (TotalSF) ",
                                                'RMSE': linreg_full_model_feature1_rmse,
                                                'R-Squared': linreg_full_model_feature1_rsquared,
                                                'Adj. R-Squared': linreg_full_model_feature1_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_feature1_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>The R-squared value of the model with new feature is slightly less than the R-squared value for linreg model with log_Property_Sale_Price. RMSE and adjusted R-squared of the model is slightly increased by introducing a new feature.  
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='feature2'></a>
# ### 5.3.2 Multiple Linear Regression (Using New Feature2) - Full Model (OLS)

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to build the model, we do the following: <br><br>
#                        1. Create two new feature by using variables 'Buiding_age' and 'Remodel_age' <br>
#                        2. Split the data into train and test sets<br>
#                        3. Build model using sm.OLS().fit()<br>
#                        4. Predict the values using test set <br>
#                        5. Compute accuracy measures <br>
#                        6. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Create two new feature by using variables 'Buiding_age' and 'Remodel_age'**

# In[ ]:


# 'datetime' is used to perform date and time operations
import datetime as dt

# 'now().year' gives the current year
# store the year as 'current_year'
current_year = int(dt.datetime.now().year)


# In[ ]:


# create 2 new variables 'Buiding_age' and 'Remoel_age' 
Buiding_age = current_year - df_house.YearBuilt
Remodel_age = current_year - df_house.YearRemodAdd


# In[ ]:


# add the new variables to the dataframe
df_house['Buiding_age'] = Buiding_age
df_house['Remodel_age'] = Remodel_age

# since we have added a new variable using the existing variables
# remove the existing variables in the data, 'YearBuilt' and 'YearRemodAdd'
# use 'drop()' to remove the variables
# 'axis = 1' drops the specific columns
df_house = df_house.drop(['YearBuilt', 'YearRemodAdd'], axis=1)


# In[ ]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# filter the categorical features in the dataset using select_dtypes()
# include=[np.object]: selects the categoric features
df_categoric_features = df_house.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_feature2 = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_full_model_feature2.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 94.9% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 1.955 and indicates that there is no autocorrelation. The Condition Number 1.28e+16 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **4. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_feature2_predictions = linreg_full_model_feature2.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_feature2_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **5. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_feature2_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_feature2_rsquared = linreg_full_model_feature2.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_feature2_rsquared_adj = linreg_full_model_feature2.rsquared_adj 


# **6. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_full_model_feature2_metrics = pd.Series({'Model': "Linreg with new features (Building_age and Remodel_age)",
                                                'RMSE': linreg_full_model_feature2_rmse,
                                                'R-Squared': linreg_full_model_feature2_rsquared,
                                                'Adj. R-Squared': linreg_full_model_feature2_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_feature2_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>RMSE of the model with new features 'Building_age' and 'Remodel_age' is increased. The value of R-squared and aadjusted R-squared is same as the previous model.  
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='feature_selection'></a>
# # 6. Feature Selection

# <a id='vif'></a>
# ## 6.1 Variance Inflation Factor
# 
# The Variance Inflation Factor (VIF) is used to detect the presence of multicollinearity between the features. The value of VIF equal to 1 indicates that no features are correlated. We calculate the VIF of the numerical independent variables. VIF for the variable V<sub>i</sub> is given as:
# <p style='text-indent:29em'> <strong> VIF = 1 / (1 - R-squared)</strong>  </p><br>
# Where, R-squared is the R-squared of the regression model build by regressing one independent variable (say V<sub>i</sub>) on all the remaining independent variables (say V<sub>j</sub>, j ≠ i).

# In[ ]:


# consider the independent variables in the dataframe 'df_house' 
# use 'drop()' to remove the target variables 'Property_Sale_Price' and 'log_Property_Sale_Price'
# 'axis = 1' drops the specific columns
df_house_features = df_house.drop(['Property_Sale_Price', 'log_Property_Sale_Price'], axis = 1)

# filter the numerical features in the dataset
df_numeric_features_vif = df_house_features.select_dtypes(include=[np.number])

# display the first five observations
df_numeric_features_vif.head()


# #### Calculate the VIF for each numeric variable.

# In[ ]:


# create an empty dataframe to store the VIF for each variable
vif = pd.DataFrame()

# calculate VIF using list comprehension 
# use for loop to access each variable 
# calculate VIF for each variable and create a column 'VIF_Factor' to store the values 
vif["VIF_Factor"] = [variance_inflation_factor(df_numeric_features_vif.values, i) for i in range(df_numeric_features_vif.shape[1])]

# create a column of variable names
vif["Features"] = df_numeric_features_vif.columns

# sort the dataframe based on the values of VIF_Factor in descending order
# 'ascending = False' sorts the data in descending order
# 'reset_index' resets the index of the dataframe
# 'drop = True' drops the previous index
vif.sort_values('VIF_Factor', ascending = False).reset_index(drop = True)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The output shows that the variable 'YrSold' has the highest VIF. Now, we use the `for loop` to find VIF and remove the variables with VIF greater than 10. We set the threshold to 10, as we wish to remove the variable for which the remaining variables explain more than 90% of the variation. One can choose the threshold  other than 10. (it depends on the business requirements)
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:


# for each numeric variable, calculate VIF and save it in a dataframe 'vif'

# use for loop to iterate the VIF function 
for ind in range(len(df_numeric_features_vif.columns)):
    
    # create an empty dataframe
    vif = pd.DataFrame()

    # calculate VIF using list comprehension
    # use for loop to access each variable 
    # calculate VIF for each variable and create a column 'VIF_Factor' to store the values 
    vif["VIF_Factor"] = [variance_inflation_factor(df_numeric_features_vif.values, i) for i in range(df_numeric_features_vif.shape[1])]

    # create a column of variable names
    vif["Features"] = df_numeric_features_vif.columns

    # filter the variables with VIF greater than 10 and store it in a dataframe 'multi' 
    # one can choose the threshold other than 10 (it depends on the business requirements)
    multi = vif[vif['VIF_Factor'] > 10]
    
    # if dataframe 'multi' is not empty, then sort the dataframe by VIF values
    # if dataframe 'multi' is empty (i.e. all VIF <= 10), then print the dataframe 'vif' and break the for loop using 'break' 
    # 'by' sorts the data using given variable(s)
    # 'ascending = False' sorts the data in descending order
    if(multi.empty == False):
        df_sorted = multi.sort_values(by = 'VIF_Factor', ascending = False)
    else:
        print(vif)
        break
    
    # use if-else to drop the variable with the highest VIF
    # if  dataframe 'df_sorted' is not empty, then drop the first entry in the column 'Features' from the numeric variables
    # select the variable using 'iloc[]'
    # 'axis=1' drops the corresponding column
    #  else print the final dataframe 'vif' with all values after removal of variables with VIF less than 10  
    if (df_sorted.empty == False):
        df_numeric_features_vif = df_numeric_features_vif.drop(df_sorted.Features.iloc[0], axis=1)
    else:
        print(vif)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The above dataframe contains all the variables with VIF less than 10. 
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to build the model, we do the following: <br><br>
#                        1. Concatenate numerical and dummy encoded categorical variables  <br>
#                        2. Split the data into train and test sets<br>
#                        3. Build model using sm.OLS().fit()<br>
#                        4. Predict the values using test set <br>
#                        5. Compute accuracy measures <br>
#                        6. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# Now, let us build the model using the categorical variables and the numerical variables obtained from VIF. 

# **1. Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# consider the variables obtained from VIF
# concatenate the numerical and dummy encoded categorical variables using concat()
# use the dummy encoded categorical variables obtained in section 5.3.4
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features_vif, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# consider independent variables
# create a copy of 'df_dummy' and store it as X
X = df_dummy.copy()

# extract the target variable from the data set
y = df_house[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_vif = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_full_model_vif.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 92.4% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 1.919 and indicates that there is no autocorrelation. The Condition Number 1.00e+20 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **4. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_vif_predictions = linreg_full_model_vif.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_vif_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **5. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_vif_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_vif_rsquared = linreg_full_model_vif.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_vif_rsquared_adj = linreg_full_model_vif.rsquared_adj 


# **6. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_full_model_vif_metrics = pd.Series({'Model': "Linreg with VIF",
                                                'RMSE': linreg_full_model_vif_rmse,
                                                'R-Squared': linreg_full_model_vif_rsquared,
                                                'Adj. R-Squared': linreg_full_model_vif_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_vif_metrics, ignore_index = True)

# print the result table
result_tabulation


# <a id='forward'></a>
# ## 6.2 Forward Selection
# 
# This method considers the null model (model with no predictors) in the first step. In the next steps start adding one variable at each step until we run out of the independent variables or the stopping rule is achieved. The variable is added based on its correlation with the target variable. Such a variable has the least p-value in the model.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to select the features using forward selection method, we do the following: <br><br>
#                        1. Concatenate numerical and dummy encoded categorical variables  <br>
#                        2. Split the data into train and test sets<br>
#                        3. Find best features using forward selection method<br>
#                        4. Build the model on the features obtained in step 3 using sm.OLS().fit()<br>
#                        5. Predict the values using test set <br>
#                        6. Compute accuracy measures <br>
#                        7. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# filter the categorical features in the dataset using select_dtypes()
# include=[np.object]: selects the categoric features
df_categoric_features = df_house.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Find best features using forward selection method**

# Consider the 'log_Property_Sale_Price' as the target variable.

# In[ ]:


# initiate linear regression model to use in feature selection
linreg = LinearRegression()

# build step forward selection
# pass the regression model to 'estimator'
# pass number of required feartures to 'k_features'. 'best' means that a best possible subset will be selected  
# 'forward=True' performs forward selection method
# 'verbose=1' returns the number of features at the corresponding step
# 'verbose=2' returns the R-squared scores and the number of features at the corresponding step
# 'scoring=r2' considers R-squared score to select the feature
# 'n_jobs = -1' considers all the CPUs in the system to select the feattures
linreg_forward = sfs(estimator = linreg, k_features = 'best', forward = True, verbose = 2, scoring = 'r2', n_jobs = -1)

# fit the step forward selection on training data using fit()
# consider the log of sales price as the target variable
sfs_forward = linreg_forward.fit(X_train, y_train['log_Property_Sale_Price'])


# In[ ]:


# print the number of selected features
print('Number of features selected using forward selection method:', len(sfs_forward.k_feature_names_))

# print a blank line
print('\n')

# print the selected feature names when k_features = 'best'
print('Features selected using forward selection method are: ')
print(sfs_forward.k_feature_names_)


# In[ ]:


df_dummy.head()


# In[ ]:


type(sfs_forward.k_feature_names_)


# In[ ]:


df_dummy_sfs = df_dummy.filter(sfs_forward.k_feature_names_)


# In[ ]:


df_dummy_sfs.head()


# In[ ]:


df_dummy_sfs.shape


# **4. Build the model on the features obtained in step 3 using sm.OLS().fit()**

# The above list shows the features selected by forward selection method. We consider all the numeric features from the list. <br>
# For categorical features, the list contains some of the categories of a categorical variable. Thus, we consider only the variables that have either all its categories in the above list or if majority of observations lie in the categories given above. To select such variables we refer to the countplot plotted in section 4.1.3

# In[ ]:


df_house.columns


# In[ ]:


abc = ['Dwell_Type', 'Zone_Class', 'Road_Type', 'LandContour', 'Neighborhood', 'HouseStyle',
                                         'OverallQual', 'OverallCond', 'RoofStyle', 'ExterQual', 'BsmtQual', 'BsmtCond', 
                                         'BsmtFinType1', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 
                                         'GarageType', 'Fence', 'SaleType', 'SaleCondition']

for i in abc:
    print(i, i in df_house)


# In[ ]:


# consider numeric features
df_numeric_features = df_house.loc[:, ['LotArea', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                                       'GarageCars', 'WoodDeckSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'YrSold',
                                       'TotalSF', 'Buiding_age', 'Remodel_age']]

# consider categoric features
df_categoric_features = df_house.loc[:, ['Dwell_Type', 'Zone_Class', 'Road_Type', 'LandContour', 'Neighborhood', 'HouseStyle',
                                         'OverallQual', 'OverallCond', 'RoofStyle', 'ExterQual', 'BsmtQual', 'BsmtCond', 
                                         'BsmtFinType1', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 
                                         'GarageType', 'Fence', 'SaleType', 'SaleCondition']]


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# In[ ]:


df_dummy.shape


# **Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# extract the target variable from the data set
y = df_house[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train_forward, X_test_forward, y_train, y_test = train_test_split(df_dummy, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train_forward is:", X_train_forward.shape)

# print dimension of predictors test set
print("The shape of X_test_forward is:",X_test_forward.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_forward = sm.OLS(y_train['log_Property_Sale_Price'], X_train_forward).fit()

# print the summary output
print(linreg_full_model_forward.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 91.6% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 2.019 and indicates that there is no autocorrelation. The Condition Number 1.30e+16 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **5. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_forward_predictions = linreg_full_model_forward.predict(X_test_forward)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_forward_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **6. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_forward_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_forward_rsquared = linreg_full_model_forward.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_forward_rsquared_adj = linreg_full_model_forward.rsquared_adj 


# **7. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_full_model_forward_metrics = pd.Series({'Model': "Linreg with Forward Selection",
                                                'RMSE': linreg_full_model_forward_rmse,
                                                'R-Squared': linreg_full_model_forward_rsquared,
                                                'Adj. R-Squared': linreg_full_model_forward_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_forward_metrics, ignore_index = True)

# print the result table
result_tabulation


# <a id="back"></a>
# ## 6.3 Backward Elimination
# 
# This method considers the full model (model with all the predictors) in the first step. In the next steps start removing one variable at each step until we run out of the independent variables or the stopping rule is achieved. The least significant variable (with the highest p-value) is removed at each step.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to select the features using backward elimination method, we do the following: <br><br>
#                        1. Concatenate numerical and dummy encoded categorical variables  <br>
#                        2. Split the data into train and test sets<br>
#                        3. Find the best features using backward selection method<br>
#                        4. Build the model on the features obtained in step 3 using sm.OLS().fit()<br>
#                        5. Predict the values using test set <br>
#                        6. Compute accuracy measures <br>
#                        7. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# filter the categorical features in the dataset using select_dtypes()
# include=[np.object]: selects the categoric features
df_categoric_features = df_house.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Find the best features using backward elimination method**

# Consider the 'log_Property_Sale_Price' as the target variable.

# In[ ]:


# initiate linear regression model to use in feature selection
linreg = LinearRegression()

# build step backward feature selection
# pass the regression model to 'estimator'
# pass number of required features to 'k_features'. 'best' means that a best possible subset will be selected 
# 'forward=False' performs backward selection method
# 'verbose=1' returns the number of features at the corresponding step
# 'verbose=2' returns the R-squared scores and the number of features at the corresponding step
# 'scoring=r2' considers R-squared score to select the feature
# 'n_jobs = -1' considers all the CPUs in the system to select the feattures
linreg_backward = sfs(estimator = linreg, k_features = 'best', forward = False, verbose = 2, scoring = 'r2', n_jobs = -1)

# fit the backward elimination on train data using fit()
# consider the log of sales price as the target variable
sfs_backward = linreg_backward.fit(X_train, y_train['log_Property_Sale_Price'])


# In[ ]:


# print the number of selected features
print('Number of features selected using backward elimination method:', len(sfs_backward.k_feature_names_))

# print a blank line
print('\n')

# print the selected feature names when k_features = 'best'
print('Features selected using backward elimination method are: ')
print(sfs_backward.k_feature_names_)


# **4. Build the model on the features obtained in step 3 using sm.OLS().fit()**

# The above list shows the features selected by backward elimination method. We consider all the numeric features from the list. <br>
# For categorical features, the list contains some of the categories of a categorical variable. Thus, we consider only the variables that have either all its categories in the above list or if majority of observations lie in the categories given above. To select such variables we refer to the countplot plotted in section 4.1.3

# In[ ]:


# consider numeric features
df_numeric_features = df_house.loc[:, ['LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                                       'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                                       'Remodel_age']]

# consider categoric features
df_categoric_features = df_house.loc[:, ['Dwell_Type', 'Zone_Class', 'LandContour', 'Neighborhood','OverallQual', 'OverallCond', 
                                         'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 
                                         'Electrical', 'KitchenQual', 'GarageType', 'GarageFinish', 'GarageQual', 'Fence',
                                         'SaleType', 'SaleCondition']]


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# In[ ]:


df_dummy.shape


# **Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# extract the target variable from the data set
y = df_house[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train_backward, X_test_backward, y_train, y_test = train_test_split(df_dummy, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train_backward is:", X_train_backward.shape)

# print dimension of predictors test set
print("The shape of X_test_backward is:",X_test_backward.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_backward = sm.OLS(y_train['log_Property_Sale_Price'], X_train_backward).fit()

# print the summary output
print(linreg_full_model_backward.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 89.0% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 2.005 and indicates that there is no autocorrelation. The Condition Number 1.30e+16 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **5. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_backward_predictions = linreg_full_model_backward.predict(X_test_backward)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_backward_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **6. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_backward_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_backward_rsquared = linreg_full_model_backward.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_backward_rsquared_adj = linreg_full_model_backward.rsquared_adj 


# **7. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_full_model_backward_metrics = pd.Series({'Model': "Linreg with Backward Elimination",
                                                'RMSE': linreg_full_model_backward_rmse,
                                                'R-Squared': linreg_full_model_backward_rsquared,
                                                'Adj. R-Squared': linreg_full_model_backward_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_backward_metrics, ignore_index = True)

# print the result table
result_tabulation


# <a id="rfe"></a>
# ## 6.4 Recursive Feature Elimination (RFE)
# 
# It is the process that returns the significant features in the dataset by recursively removing the less significant feature subsets.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In order to select the features using RFE method, we do the following: <br><br>
#                        1. Concatenate numerical and dummy encoded categorical variables  <br>
#                        2. Split the data into train and test sets<br>
#                        3. Find the best features using RFE method<br>
#                        4. Build the model on the features obtained in step 3 using sm.OLS().fit()<br>
#                        5. Predict the values using test set <br>
#                        6. Compute accuracy measures <br>
#                        7. Tabulate the results
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number: selects the numeric features
df_numeric_features = df_house.select_dtypes(include=np.number)

# filter the categorical features in the dataset using select_dtypes()
# include=[np.object]: selects the categoric features
df_categoric_features = df_house.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Find the best features using RFE method**

# Consider the 'log_Property_Sale_Price' as the target variable.

# In[ ]:


# initiate linear regression model to use in feature selection
linreg_rfe = LinearRegression()

# build the RFE model
# pass the regression model to 'estimator'
# if we do not pass the number of features, RFE considers half of the features
rfe_model = RFE(estimator = linreg_rfe)

# fit the RFE model on train data using fit()
# consider the log of sales price as the target variable
rfe_model = rfe_model.fit(X_train, y_train['log_Property_Sale_Price'])

# create a series containing feature and its corresponding rank obtained from RFE
# 'ranking_' returns the rank of each variable after applying RFE
# pass the ranks as the 'data' of a series
# 'index' assigns feature names as index of a series 
feat_index = pd.Series(data = rfe_model.ranking_, index = X_train.columns)

# select the features with rank = 1
# 'index' returns the indices of a series (i.e. features with rank=1) 
signi_feat_rfe = feat_index[feat_index==1].index

# print the number of selected features 
print('Number of features selected using RFE method:', len(signi_feat_rfe))

# print a blank line
print('\n')

# print the significant features obtained from RFE
print('Features selected using RFE method are:')
print(list(signi_feat_rfe))


# **4. Build the model on the features obtained in step 3 using sm.OLS().fit()**

# The above list shows the features selected by RFE method. We consider all the numeric features from the list. <br>
# For categorical features, the list contains some of the categories of a categorical variable. Thus, we consider only the variables that have either all its categories in the above list or if majority of observations lie in the categories given above. To select such variables we refer to the countplot plotted in section 4.1.3

# In[ ]:


# consider numeric features
df_numeric_features = df_house.loc[:, ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces', 
                                       'GarageCars']]

# consider categoric features
df_categoric_features = df_house.loc[:, ['Dwell_Type', 'Zone_Class', 'Road_Type', 'Neighborhood','OverallQual', 'OverallCond',
                                         'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'BsmtFinType1', 'Heating', 
                                         'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageQual',
                                         'GarageCond', 'SaleType', 'SaleCondition']]


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# In[ ]:


df_dummy.shape


# **Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# extract the target variable from the data set
y = df_house[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train subset and test subset for predictor and target variables
# 'test_size' returns the proportion of data to be included in the test set
# set 'random_state' to generate the same dataset each time you run the code 
X_train_rfe, X_test_rfe, y_train, y_test = train_test_split(df_dummy, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train_rfe is:", X_train_rfe.shape)

# print dimension of predictors test set
print("The shape of X_test_rfe is:",X_test_rfe.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full_model_rfe = sm.OLS(y_train['log_Property_Sale_Price'], X_train_rfe).fit()

# print the summary output
print(linreg_full_model_rfe.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains 89.0% of the variation in dependent variable log_Property_Sale_Price. The Durbin-Watson test statistics is 1.995 and indicates that there is no autocorrelation. The Condition Number 1.27e+16 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **5. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_full_model_rfe_predictions = linreg_full_model_rfe.predict(X_test_rfe)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_full_model_rfe_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **6. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_full_model_rfe_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_full_model_rfe_rsquared = linreg_full_model_rfe.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_rfe_rsquared_adj = linreg_full_model_rfe.rsquared_adj 


# **7. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_full_model_rfe_metrics = pd.Series({'Model': "Linreg with RFE",
                                                'RMSE': linreg_full_model_rfe_rmse,
                                                'R-Squared': linreg_full_model_rfe_rsquared,
                                                'Adj. R-Squared': linreg_full_model_rfe_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_model_rfe_metrics, ignore_index = True)

# print the result table
result_tabulation


# <a id="conclusion"> </a>
# # 7. Conclusion and Interpretation

# To take the final conclusion, let us print the result table.

# In[ ]:


# print the 'result_tabulation' to compare all the models
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The regression algorithms named in the above table have been implemented on the givn dataset. The performance of the models were evaluated using RMSE, R-squared and Adjusted R-squared values. <br><br>
#                         The above result shows that the linear regression with forward selection method has the lowest RMSE value. Thus, it can be concluded that the linear regression model build on the features obtained from forward selection method can be used to predict the price of the house.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:


a = [1,2,3]
b = [3,4,5]
zip(a,b)


# In[ ]:




