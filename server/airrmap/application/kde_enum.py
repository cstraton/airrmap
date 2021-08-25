# %% Enums for the kde module
# Stored in a separate module to avoid circular references.
# e.g. used by TileRequest and KDEManager.

from enum import Enum, unique

@unique
class KDERelativeMode(Enum):
    """
    Determines the population for computing relative KDE relative values.

    Parameters
    ----------
    Enum : int

        SINGLE: Values are computed relative to another, single KDE,
            or values are -not- computed relative to other KDEs (i.e. uses self).

        ROW: Values are computed relative to a row's values.

        COLUMN: Values are computed relative to a column's values.


        ALL: Values are computed relative to all KDEs.
    """
    SINGLE = 0
    ROW = 1
    COLUMN = 2
    ALL = 3


@unique
class KDERelativeSelection(Enum):
    """
    Determines the selection within the reference population defined by KDERelativeMode.

    KDE values will be computed relative to the selection here.

    Parameters
    ----------
    Enum : int
        ALL: All KDEs within the population.

        FIRST: The first KDE within the population. Only valid where KDERelativeMode is
            ROW, COLUMN or ALL.

        LAST: The last KDE within the population. Only valid where KDERelativeMode is
            ROW, COLUMN or ALL.

        PREVIOUS: The previous KDE within the population, relative to the KDE
            being computed. Only valid if KDERelativeMode is ROW or COLUMN.
    """
    ALL = 0
    FIRST = 1
    LAST = 2
    PREVIOUS = 4