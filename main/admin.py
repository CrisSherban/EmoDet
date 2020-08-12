from django.contrib import admin
from .models import Person

# Register your models here.
class PersonAdmin(admin.ModelAdmin):
	fields = [	"person_emotion", 
				"person_number",
			  	"person_last_seen"]
	

admin.site.register(Person, PersonAdmin)