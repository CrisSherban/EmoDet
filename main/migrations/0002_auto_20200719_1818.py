# Generated by Django 3.0.8 on 2020-07-19 18:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='person',
            old_name='person_last_seet',
            new_name='person_last_seen',
        ),
    ]
