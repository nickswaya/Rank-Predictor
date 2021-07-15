from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, validators
from wtforms.validators import DataRequired

class SubmitReplay(FlaskForm):
    replay_id = StringField('Replay ID', validators=[DataRequired(), validators.Length(36, message='Match ID not valid: Length Error')])
    submit = SubmitField('Submit Replay ID')