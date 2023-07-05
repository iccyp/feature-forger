import pandas as pd

from feature_forger.domain.entities.pipeline import Pipeline


class PipelineRunner:

    def run(self, pipeline: Pipeline, copy: bool) -> pd.DataFrame:
        entity_rows_df = pipeline.dataset.map_rows_to_entity(
            pipeline.entity)
        input_data = entity_rows_df.copy() if copy else entity_rows_df
        result = pipeline.flow(input_data)
        return result
