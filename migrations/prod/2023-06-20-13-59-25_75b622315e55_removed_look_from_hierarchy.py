"""removed look from hierarchy

Revision ID: 75b622315e55
Revises: 764a8c3bb569
Create Date: 2023-06-20 13:59:25.470259

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '75b622315e55'
down_revision = '764a8c3bb569'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('styles', sa.Column('story', sa.String(), nullable=False))
    op.drop_column('styles', 'look')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('styles', sa.Column('look', sa.VARCHAR(), autoincrement=False, nullable=False))
    op.drop_column('styles', 'story')
    # ### end Alembic commands ###
