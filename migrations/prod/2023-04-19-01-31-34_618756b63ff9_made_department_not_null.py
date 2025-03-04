"""made department not null

Revision ID: 618756b63ff9
Revises: 733920e14c83
Create Date: 2023-04-19 01:31:34.821806

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '618756b63ff9'
down_revision = '733920e14c83'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('role', sa.Column('department_id', sa.Integer(), nullable=False))
    op.alter_column('role', 'feature',
                    existing_type=sa.VARCHAR(length=255),
                    nullable=False)
    op.drop_constraint('role_department_fkey', 'role', type_='foreignkey')
    op.create_foreign_key(None, 'role', 'department', ['department_id'], ['id'], ondelete='CASCADE')
    op.drop_column('role', 'department')
    op.add_column('user', sa.Column('department_id', sa.Integer(), nullable=True))
    op.drop_constraint('user_department_fkey', 'user', type_='foreignkey')
    op.create_foreign_key(None, 'user', 'department', ['department_id'], ['id'], ondelete='CASCADE')
    op.drop_column('user', 'department')
    op.drop_column('user', 'name')
    op.drop_column('user', 'pic')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('user', sa.Column('pic', sa.VARCHAR(length=255), autoincrement=False, nullable=True))
    op.add_column('user', sa.Column('name', sa.VARCHAR(length=255), autoincrement=False, nullable=False))
    op.add_column('user', sa.Column('department', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_constraint(None, 'user', type_='foreignkey')
    op.create_foreign_key('user_department_fkey', 'user', 'department', ['department'], ['id'], ondelete='CASCADE')
    op.drop_column('user', 'department_id')
    op.add_column('role', sa.Column('department', sa.INTEGER(), autoincrement=False, nullable=True))
    op.drop_constraint(None, 'role', type_='foreignkey')
    op.create_foreign_key('role_department_fkey', 'role', 'department', ['department'], ['id'], ondelete='CASCADE')
    op.alter_column('role', 'feature',
                    existing_type=sa.VARCHAR(length=255),
                    nullable=True)
    op.drop_column('role', 'department_id')
    # ### end Alembic commands ###
