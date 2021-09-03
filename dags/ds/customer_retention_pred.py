from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from customer_retention.prod.xgboost.customer_retention_pred import CustomerRetentionPred
import mlflow
from datetime import date
import os

def customer_retention_pred(**kwargs):
    mlflow.set_tracking_uri(os.environ.get('MLFLOW__CORE__SQL_ALCHEMY_CONN'))
    visits_number = [1, 2, 3, 4, 5]
    visits_name = ['visit_' + str(x) for x in visits_number]
    for v_num, v_name in zip(visits_number, visits_name):
        mlflow.set_experiment(v_name)
        cr = CustomerRetentionPred()
        cr.start(visit_number=v_num)
    return "Predict has run succesfully"

# Airflow Email Operators
def success_email_operator(**context):
    # xcom_path = context['ti'].xcom_pull(key='error_path')
    # xcom_subject = context['ti'].xcom_pull(key='success_subject')
    op = EmailOperator(
        task_id="success_email_operator",
        to=['asuliman@Coldborecapital.com'],
        subject="succesful run",
        html_content="""Try {{try_number}} out of {{max_tries + 1}}<br>
                        Exception:<br>{{exception_html}}<br>
                        Log: <a href="{{ti.log_url}}">Link</a><br>
                        Host: {{ti.hostname}}<br>
                        Log file: {{ti.log_filepath}}<br>
                        Mark success: <a href="{{ti.mark_success_url}}">Link</a><br>""")
    op.execute(context)


def failed_email_operator(**context):
    # xcom_path = context['ti'].xcom_pull(key='error_path')
    xcom_subject = context['ti'].xcom_pull(key='error_subject')
    op = EmailOperator(
        task_id="failure_email_operator",
        to=['asuliman@Coldborecapital.com'],
        subject=xcom_subject,
        #    files=[xcom_path,],
        html_content="""Try {{try_number}} out of {{max_tries + 1}}<br>
                        Exception:<br>{{exception_html}}<br>
                        Log: <a href="{{ti.log_url}}">Link</a><br>
                        Host: {{ti.hostname}}<br>
                        Log file: {{ti.log_filepath}}<br>
                        Mark success: <a href="{{ti.mark_success_url}}">Link</a><br>""")
    op.execute(context)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': 'asuliman@coldborecapital.com',
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'on_success_callback': success_email_operator,
    'on_failure_callback': failed_email_operator
}

dag = DAG(
    "Model_Predict",
    description="Predict every hour",
    schedule_interval='0 * * * *',
    default_args=default_args,
    catchup=False,
)

t1 = PythonOperator(task_id="CustomerRetentionPred", python_callable=customer_retention_pred, dag=dag,
                    provide_context=True)

t1
