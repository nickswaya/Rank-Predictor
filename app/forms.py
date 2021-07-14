from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class SubmitReplay(FlaskForm):
    replay_id = StringField('Replay ID', validators=[DataRequired()])
    submit = SubmitField('Submit Replay ID')