{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
      {% if attributes %}
         .. rubric:: Module Attributes

         .. autosummary::
            :toctree:
            {% for item in attributes %}
               {{ item }}
            {%- endfor %}
      {% endif %}
   {% endblock %}

   {% block functions %}
      {% if functions %}
         .. rubric:: {{ _('Functions') }}

         .. autosummary::
            :nosignatures:
            {% for item in functions %}
               .. autofunction:: {{ item }}
            {%- endfor %}
      {% endif %}
   {% endblock %}

   {% block classes %}
      {% if classes %}
         .. rubric:: {{ _('Classes') }}

         .. autosummary::
            :nosignatures:
            {% for item in classes %}
            .. autoclass:: {{ objname }}
               :members:
               :show-inheritance:
               :inherited-members:
               {% block methods %}
                  {% if methods %}
                     .. rubric:: {{ _('Methods') }}
                     .. autosummary::
                        {% for item in methods %}
                           ~{{ name }}.{{ item }}
                        {%- endfor %}
                  {% endif %}
               {% endblock %}
            {%- endfor %}
      {% endif %}
   {% endblock %}

   {% block exceptions %}
      {% if exceptions %}
         .. rubric:: {{ _('Exceptions') }}

         .. autosummary::
            :nosignatures:
            {% for item in exceptions %}
               .. autoclass:: {{ item }}
                  :members:
                  :show-inheritance:
            {%- endfor %}
      {% endif %}
   {% endblock %}

{% block modules %}
   {% if modules %}
      .. rubric:: Modules

      .. autosummary::
         :toctree:
         :template: module.rst
         :recursive:
         {% for item in modules %}
            {{ item }}
         {%- endfor %}
   {% endif %}
{% endblock %}
