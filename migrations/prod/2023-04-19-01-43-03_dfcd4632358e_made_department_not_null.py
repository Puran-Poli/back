"""made department not null

Revision ID: dfcd4632358e
Revises: d7ad0fc65928
Create Date: 2023-04-19 01:43:03.888652

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'dfcd4632358e'
down_revision = 'd7ad0fc65928'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('user', 'department_id',
                    existing_type=sa.INTEGER(),
                    nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('user', 'department_id',
                    existing_type=sa.INTEGER(),
                    nullable=True)
    # ### end Alembic commands ###
