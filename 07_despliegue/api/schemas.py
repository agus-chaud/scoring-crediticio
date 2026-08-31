from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RegistroEntrada(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id_cliente: Optional[int] = None
    ingresos_verificados: str
    vivienda: str
    finalidad: str
    num_cuotas: str
    antiguedad_empleo: str = Field(alias="antigüedad_empleo")
    rating: str
    ingresos: float
    dti: float
    num_lineas_credito: float
    porc_uso_revolving: float
    principal: float
    tipo_interes: float
    imp_cuota: float
    num_derogatorios: float


class ScoringSalida(BaseModel):
    id_cliente: Optional[int] = None
    score_pd: float
    score_ead: float
    score_lgd: float
    perdida_esperada_relativa: float
