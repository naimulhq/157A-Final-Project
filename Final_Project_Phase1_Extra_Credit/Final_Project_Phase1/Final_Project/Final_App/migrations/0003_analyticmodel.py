# Generated by Django 3.1.3 on 2020-12-11 01:17

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('Final_App', '0002_algorithmmodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='AnalyticModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('analytic_name', models.CharField(max_length=50)),
                ('result_plot', models.CharField(max_length=10000000)),
                ('time', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
    ]