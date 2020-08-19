# Generated by Django 3.0.8 on 2020-07-30 18:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0007_auto_20200729_1821'),
    ]

    operations = [
        migrations.CreateModel(
            name='OverallStats',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_avg_emotion', models.CharField(max_length=20, null=True)),
                ('avg_emotion_plot', models.ImageField(blank=True, null=True, upload_to='faces_pics/')),
            ],
        ),
    ]