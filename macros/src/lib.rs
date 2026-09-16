//! Macros procédurales de pyrucast.
//!
//! Une seule pour l'instant, à venir : `#[py_op]`, qui dérive d'une fonction
//! libre exposée à Python la **méthode** de son sujet (`CONVENTIONS.md`, § « Le
//! verbe exposé aussi en méthode »).
//!
//! Pourquoi une macro procédurale et pas une `macro_rules!` : il faut *lire*
//! une signature déjà écrite pour en retirer le premier paramètre, et
//! transformer `#[pyo3(signature = (mesh, angle_deg=None))]` en la même liste
//! privée de sa première entrée. Une `macro_rules!` ne voit que les jetons
//! qu'on lui passe et ne sait pas ouvrir un fragment capturé ; une macro
//! procédurale reçoit l'item entier et le décompose.
//!
//! Elle recopie aussi les `///` de la fonction en littéraux sur la méthode :
//! c'est ce qui met la documentation complète à la fois dans `__doc__` et dans
//! le stub `.pyi` que lisent les IDE.
