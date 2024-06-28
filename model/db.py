#db.py
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from typing import AsyncGenerator

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from config import logger, settings

Base = declarative_base()
NotNullColumn = partial(Column, nullable=False)


class Styles(Base):
    __tablename__ = "styles"

    sizes = relationship("Sizes", backref="style", lazy="joined")

    product_id = NotNullColumn(String, primary_key=True, index=True)
    style_id = NotNullColumn(String)

    gender = NotNullColumn(String)
    vertical = NotNullColumn(String)
    fabric_category = NotNullColumn(String)
    brand = NotNullColumn(String)
    usage = NotNullColumn(String)
    brick = NotNullColumn(String)
    product = NotNullColumn(String)
    sub_product = NotNullColumn(String)
    target_audience = NotNullColumn(String)
    fit = NotNullColumn(String)

    mrp = Column(Integer)
    cost = Column(Integer)

    story = NotNullColumn(String)
    tags = Column(String)

    colour_family = NotNullColumn(String)
    primary_colour = NotNullColumn(String)
    secondary_colour = Column(String)
    tertiary_colour = Column(String)

    season = Column(String)
    exclusive = NotNullColumn(String)

    garment_pattern = Column(String)
    print_pattern_type = Column(String)
    number_of_components = Column(String)
    number_of_pockets = Column(Integer)
    pocket_type = Column(String)
    neck = Column(String)
    collar = Column(String)
    placket = Column(String)
    length = Column(String)
    sleeve_length = Column(String)
    sleeve_type = Column(String)
    hemline = Column(String)
    waist_rise = Column(String)
    closure = Column(String)
    footwear_ankle_type = Column(String)
    footwear_insole = Column(String)

    fabric_code = Column(String)
    fabric_rate = Column(Integer)
    fabric_story = Column(String)
    fabric_composition = Column(String)
    fabric_hsn = Column(Integer)
    fabric_weave_pattern = Column(String)
    fabric_vendor = Column(String)
    denim_cast = Column(String)
    denim_wash = Column(String)
    wash_care = Column(String)
    footwear_upper_material = Column(String)
    footwear_sole_material = Column(String)

    first_grn_date = Column(Date)
    first_live_date = Column(Date)
    first_sold_date = Column(Date)

    images = Column(Text)
    

    status = NotNullColumn(String, default="Design")
    created_at = NotNullColumn(DateTime, default=datetime.now)
    created_by = NotNullColumn(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    updated_at = Column(DateTime, default=datetime.now)
    updated_by = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))

    deleted = NotNullColumn(Boolean, default=False)
    product_description = Column(String, nullable=True)  # Add this line
    dress_type = Column(String) #new

    top_style_type = Column(String)
    jacket_type = Column(String)
    distress = Column(String)
    fade = Column(String)
    dungaree_bottom_type = Column(String)





class Sizes(Base):
    __tablename__ = "sizes"

    product_id = NotNullColumn(
        String, ForeignKey("styles.product_id", ondelete="CASCADE"), primary_key=True
    )
    standard_size = NotNullColumn(String, primary_key=True)
    barcode_size = NotNullColumn(String)

    ean = Column(BigInteger, unique=True)
    myntra_id = Column(BigInteger)
    ajio_id = Column(BigInteger)
    ajio_ean = Column(BigInteger)

    garment_waist = Column(Float)
    inseam_length = Column(Float)
    to_fit_waist = Column(Float)
    across_shoulder = Column(Float)
    chest = Column(Float)
    front_length = Column(Float)
    to_fit_foot_length = Column(Float)
    shoe_weight = Column(String)
    outseam_length = Column(Float)
    bust = Column(Float)
    to_fit_bust = Column(Float)
    hip = Column(Float)


class User(Base):
    __tablename__ = "user"
    id = NotNullColumn(Integer, primary_key=True, autoincrement=True)
    email = NotNullColumn(String(255), unique=True)
    department_id = NotNullColumn(
        Integer, ForeignKey("department.id", ondelete="CASCADE")
    )
    department = relationship("Department", backref="user", lazy="joined")
    deleted = NotNullColumn(Boolean, default=False)


class Department(Base):
    __tablename__ = "department"
    # users = relationship("User", backref="department_ref", lazy="joined")
    roles = relationship("Role", backref="department_ref", lazy="joined")
    id = NotNullColumn(Integer, primary_key=True, autoincrement=True)
    name = NotNullColumn(String(255), unique=True)


class Role(Base):
    __tablename__ = "role"
    id = NotNullColumn(Integer, primary_key=True, index=True)
    department_id = NotNullColumn(
        Integer, ForeignKey("department.id", ondelete="CASCADE")
    )
    feature = NotNullColumn(String(255))
    create = NotNullColumn(Boolean(), default=False)
    read = NotNullColumn(Boolean(), default=False)
    update = NotNullColumn(Boolean(), default=False)
    delete = NotNullColumn(Boolean(), default=False)


class Session(Base):
    __tablename__ = "session"
    id = NotNullColumn(Integer, primary_key=True, autoincrement=True)
    user_id = NotNullColumn(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    login_time = NotNullColumn(DateTime)
    deleted = NotNullColumn(Boolean(), default=False)


class Channel(Base):
    __tablename__ = "channel"
    id = NotNullColumn(Integer, primary_key=True)
    name = NotNullColumn(String(255), unique=True)


class WIP(Base):
    __tablename__ = "wip"
    date = NotNullColumn(Date, primary_key=True)
    ean = NotNullColumn(BigInteger, primary_key=True)
    qty = NotNullColumn(Integer)


class GRN(Base):
    __tablename__ = "grn"
    date = NotNullColumn(Date, primary_key=True)
    ean = NotNullColumn(BigInteger, primary_key=True)
    qty = NotNullColumn(Integer)


class SOH(Base):
    __tablename__ = "soh"
    channel_id = NotNullColumn(
        Integer, ForeignKey("channel.id", ondelete="CASCADE"), primary_key=True
    )
    date = NotNullColumn(Date, primary_key=True)
    ean = NotNullColumn(BigInteger, primary_key=True)
    qty = NotNullColumn(Integer)


class Sales(Base):
    __tablename__ = "sales"
    channel_id = NotNullColumn(
        Integer, ForeignKey("channel.id", ondelete="CASCADE"), primary_key=True
    )
    date = NotNullColumn(Date, primary_key=True)
    ean = NotNullColumn(BigInteger, primary_key=True)
    qty = NotNullColumn(Integer)
    sp_per_pc = NotNullColumn(Float)
    value = NotNullColumn(Float)


class Returns(Base):
    __tablename__ = "returns"
    channel_id = NotNullColumn(
        Integer, ForeignKey("channel.id", ondelete="CASCADE"), primary_key=True
    )
    date = NotNullColumn(Date, primary_key=True)
    ean = NotNullColumn(BigInteger, primary_key=True)
    qty = NotNullColumn(Integer)

class Tag(Base):
    __tablename__ = 'tag'  # Name of the table in the database
    tag_name = NotNullColumn(String(255), primary_key=True)  # Primary key column and tag name
    tag_url_column = Column(Text)  



'''
@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        async with session.begin():
            try:
                # logger.debug("Session begun...")
                yield session
                await session.commit()
                # logger.debug("Session commited...")
            except DBAPIError as ex:
                await session.rollback()
                logger.debug("Session timeout...")
                raise ex
            except SQLAlchemyError as ex:
                await session.rollback()
                logger.debug("Session rollback...")
                raise ex
            finally:
                await session.close()
                # logger.debug("Session closed...")

'''
@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            # Explicitly begin a transaction
            await session.begin()
            yield session
        except Exception as ex:
            # Log and handle any exceptions that occur during transaction
            logger.error(f"Error during database operation: {ex}")
            await session.rollback()  # Roll back on exceptions
            logger.debug("Transaction rolled back due to an error.")
        else:
            # If no exceptions, commit the transaction
            await session.commit()
            logger.debug("Transaction committed successfully.")
        finally:
            # Close the session after operation
            await session.close()
            logger.debug("Session closed.")

engine = create_async_engine(
    settings.asyncpg_url,
    future=True,
    echo=True,
    connect_args={"server_settings": {"statement_timeout": "100000"}},
)
async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
)
