from dependency_injector import containers, providers

from feature_forger.domain.container import DomainContainer


class Container(containers.DeclarativeContainer):

    domain = providers.Container(
        DomainContainer
    )
