import streamlit as st
import pandas as pd
import base64
import numpy as np
import os
import ydata_profiling
from nbformat import read
import warnings
warnings.filterwarnings('ignore')

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def main():
	"""Generates interactive exploratory data analysis in your browser with streamlit 
	automatically cleans and interpolate the dataset with high accuracy based on the data and returns a python script with the cleaned dataset"""
	st.title('ETL-PM streamlit app to properly clean datasets')
	st.write("""
		# Interactive Data Exploration and Manual data cleaning and Data pre-processing 
		Using pandas-profiling to explore the data at the click of one button. 
	""")

	html_temp = """
		<div style="background-color:tomato;"><p style="color:white;font-size:21px;">Exploring data with pandas_profiler </p></div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)
	st.write(' ')
	st.write('*To start generating a data analysis and cleanning to clean small and large data click:*')
	st.write(' ')
	st.write('To start go to the sidebar and click on the pandas-profiling button')
	st.write(' ')
	st.write(' ')
	st.write(' ')
	st.write('*To clean and prepare the dataset for ML and to download it scroll down to the bottom of the page, you will see a tab with a button to Download the data automatically cleaned, you can also view the python script that was used to clean the data*')
	st.write(' ')
	st.write(' ')
	st.write(' *You can also start a fresh cleaning process by loading another dataset by clicking on the Load dataset button on the sidebar*')


@st.cache
def load_data(dataset):
	df=pd.read_csv(dataset)
	df.head()
	return df

def load_data_columns(dataset):
	df=pd.read_csv(dataset)
	column_names=df.columns
	return column_names

def load_data_card(dataset):
	df=pd.read_csv(dataset)
	column_names=df.columns
	options=list(column_names)
	return options

def load_data_main(dataset):
	df=pd.read_csv(dataset)
	df.head()
	return df

def load_data_nanvalues(dataset):
	df=pd.read_csv(dataset)
	column_names=df.columns
	options=list(column_names)
	return options


def load_data_datatypes(dataset):
	df=pd.read_csv(dataset)
	column_names=df.columns
	options=list(column_names)
	return options


def load_data_duplicated(dataset):
	df=pd.read_csv(dataset)
	column_names=df.columns
	options=list(column_names)
	return options

def load_data_uniques(dataset):
	df=pd.read_csv(dataset)
	column_names=df.columns
	options=list(column_names)
	return options




def main_pandas_profiling(df):
	profile = df.profile_report(title='Data exploration and cleaning report')
	st.write(profile)
	st.write(' ')
	st.write(' ')
	st.write('After reviewing your data to clean and prepare it further scroll to the bottom of the page to view the cleaned dataset')


def load_code():
	read_nbnotebook=pd.read_csv('Python_script_to_clean_data.csv')
	return read_nbnotebook


def main_cleaning_data(df,column_names):
	
	if st.sidebar.checkbox('Nan values'):
		st.subheader('Nan values')
		nan_values=st.multiselect("Select column with nan values:", options=column_names, default=column_names[0])
		values=st.text_input('Enter the values to put in place of the nan values:')
		for value in nan_values:
			df[value].fillna(values, inplace=True)
		st.subheader('Nan values automatically removed from the dataset')
		st.write(df)
		st.write("""
			# Note:
			This dataset is now ready to be downloaded, you can also download the python script used above to clean the data
		""")
		st.subheader('The Python script used to clean the data above:')
		for value in nan_values:
			st.write("df['"+value+"'].fillna("+values+", inplace=True)")

	if st.sidebar.checkbox('Data types'):
		st.subheader('Data types')
		data_type=st.selectbox("Choose the column to change the data type:", options=column_names, index=0)
		dtypes=st.selectbox("Choose the data type:", options=('float64','int64','object','datetime64','bool','category','timedelta[ns]'), index=0)
		df[data_type]=df[data_type].astype(dtypes)
		st.subheader('The column data type was successfully changed')
		st.write(df)
		st.subheader('Below is the python script used to change the data type:')
		st.write("df['"+data_type+"']=df['"+data_type+"'].astype('"+dtypes+"')")

	if st.sidebar.checkbox('Remove extra whitespace'):
		st.subheader('Remove extra whitespace')
		extra_whitespace=st.selectbox("Choose the column to remove white space", options=column_names, index=0)
		df[extra_whitespace]=df[extra_whitespace].str.strip()
		st.subheader('Extra white space successfully removed from the dataset')
		st.write(df)
		st.subheader('Below is the python script used to remove the whitespace:')
		st.write("df['"+extra_whitespace+"']=df['"+extra_whitespace+"'].str.strip()")

	if st.sidebar.checkbox('Delete columns'):
		st.subheader('Delete columns')
		delete_column=st.multiselect("Choose the columns to delete:", options=column_names, default=column_names[0])
		df=df.drop(delete_column, axis=1)
		st.subheader('Column successfully deleted from the dataset')
		st.write(df)
		st.subheader('Below is the python script used to delete columns:')
		for value in delete_column:
			st.write("df=df.drop('"+value+"', axis=1)")

	if st.sidebar.checkbox('Duplicated data'):
		st.subheader('Duplicated data')
		duplicated_data=st.multiselect("Choose duplicated data", options=column_names, default=column_names[0])
		df=df.drop_duplicates(subset=duplicated_data)
		st.subheader('Duplicated values automatically removed from the dataset')
		st.write(df)
		st.subheader('Below is the python script used to clear duplicated values:')
		for value in duplicated_data:
			st.write("df=df.drop_duplicates(subset='"+value+"')")

	if st.sidebar.checkbox('Uniques data'):
		st.subheader('Uniques data')
		uniques_data=st.text_input('Enter a value to filter out unique values:')
		df=df.drop_duplicates(subset=uniques_data)
		st.subheader('Duplicated values automatically removed from the dataset')
		st.write(df)
		st.subheader('Below is the python script used to clear duplicated values:')
		st.write("df=df.drop_duplicates(subset='"+uniques_data+"')")

	if st.sidebar.checkbox('Reset index'):
		st.subheader('Reset index')
		df=df.reset_index(drop=True)
		st.subheader('Index was successfully reset')
		st.write(df)
		st.subheader('Below is the python script used to reset the index:')
		st.write("df=df.reset_index(drop=True)")

	if st.sidebar.checkbox('Delete outliers'):
		st.subheader('Delete outliers')
		delete_outliers=st.multiselect('Choose the columns to delete outliers:', options=column_names, default=column_names[0])
		df=df[(np.abs(df[delete_outliers]-df[delete_outliers].mean())<=(3*df[delete_outliers].std()))]
		st.subheader('Outliers successfully deleted from the dataset')
		st.write(df)
		st.subheader('Below is the python script used to delete outliers:')
		for value in delete_outliers:
			st.write("df=df[(np.abs(df['"+value+"']-df['"+value+"'].mean())<=(3*df['"+value+"'].std()))]")

	if st.sidebar.checkbox('Interpolate values'):
		st.subheader('Interpolate values')
		interpolate_data=st.multiselect('Choose the columns to interpolate values:', options=column_names, default= column_names[0])
		method=st.selectbox('Choose the interpolation method:', options=('linear','time','index','values','nearest','zero','slinear','quadratic','cubic','barycentric','polynomial','krogh','piecewise_polynomial','spline','pchip','akima'), index=0)
		df[interpolate_data]=df[interpolate_data].interpolate(method=method)
		st.subheader('Interpolate values successfully interpolated to the dataset')
		st.write(df)
		st.subheader('Below is the python script used to remove outliers:')
		for value in interpolate_data:
			st.write("df['"+value+"']=df['"+value+"'].interpolate(method='"+method+"')")

	if st.sidebar.checkbox('Cleaning dataset'):
		st.subheader('Cleaning dataset')
		df.to_csv('Clean_dataset.csv')
		st.write("""
	        """ + df.to_string())
		st.write("""
			# Note:
			This dataset is now ready to be downloaded, you can also download the python script used above to clean the data
		""")
	return df




if __name__ == '__main__':
	html_title = """
		<div style="background-color:tomato;padding:2px">
		<h2 style="color:white;text-align:center;">Welcome to ETL-PM</h2>
		</div>
		"""
	st.markdown(html_title, unsafe_allow_html=True)
	dataset=load_data_columns(file_selector())
	st.markdown("""
		# Click on the button below to generate a data analysis and 
		clean the data with the buttons on the sidebar
		""")
	st.markdown("""
			# Download the dataset after cleaning and processing.
			""")
	not_cleaned=load_data_main(file_selector())
	clean_df=main_cleaning_data(not_cleaned,load_data_datatypes(file_selector()))
	st.write(clean_df)
	script=load_code()
	st.dataframe(script)

	st.markdown("""
			# You can also download the cleaned dataset.
			""")
	main()