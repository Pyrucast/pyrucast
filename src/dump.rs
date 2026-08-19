//! Third display level: **content**.
//!
//! The crate exposes three, layered display levels (see the crate root doc):
//!
//! | Level     | Python     | Role                                            | Bound        |
//! |-----------|------------|-------------------------------------------------|--------------|
//! | `Display` | `__str__`  | one line: identity + key dimensions             | O(1)         |
//! | `Debug`   | `__repr__` | structure: counts, dimensions, names, handles   | bounded      |
//! | [`Dump`]  | `dump(…)`  | **full content**: grids, value tables, topology | [`DumpOptions`] |
//!
//! `Display`/`Debug` never print bulk content; [`Dump::dump`] does (straight to
//! stdout), but stays bounded by [`DumpOptions`] (precision + row/column
//! elision). Implementors provide [`Dump::render`] (the String core, used for
//! composition); [`Dump::dump`] prints it.
//!
//! # Example
//!
//! ```
//! use pyrucast::dump::{Dump, DumpOptions};
//!
//! struct Pair(f64, f64);
//! impl Dump for Pair {
//!     fn render(&self, o: &DumpOptions) -> String {
//!         format!("({:.*}, {:.*})", o.precision, self.0, o.precision, self.1)
//!     }
//! }
//! assert_eq!(Pair(1.5, 2.0).render(&DumpOptions::default()), "(1.500, 2.000)");
//! Pair(1.5, 2.0).dump(); // prints "(1.500, 2.000)" to stdout
//! ```

/// Knobs controlling how much content [`Dump::render`] emits.
///
/// Defaults: `precision = 3`, `max_rows = 20`, `max_cols = 12`. Beyond the row
/// / column caps the renderers elide and append a `… (N de plus)` marker, so a
/// dump never grows without bound on a large object.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::dump::{self, Dump, DumpOptions};
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let mesh = Mesh::from_submesh(sm);
/// // Précision, et deux plafonds au-delà desquels les rendus élident et
/// // ajoutent un « … (N de plus) » : un dump ne grossit jamais sans borne.
/// let d = DumpOptions::default();
/// assert_eq!((d.precision, d.max_rows, d.max_cols), (3, 20, 12));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DumpOptions {
    /// Digits after the decimal point for floating-point values.
    pub precision: usize,
    /// Maximum number of data rows rendered before eliding the rest.
    pub max_rows: usize,
    /// Maximum number of value columns (column 0, the label column, is always
    /// kept) rendered before eliding the rest.
    pub max_cols: usize,
}

impl Default for DumpOptions {
    fn default() -> Self {
        Self {
            precision: 3,
            max_rows: 20,
            max_cols: 12,
        }
    }
}

/// Human-readable, bounded dump of an object's **full content**.
///
/// Implementors provide [`render`](Dump::render) — the actual numbers /
/// topology (matrix grids, field value tables, mesh connectivity) as a
/// `String`. The user-facing [`dump`](Dump::dump) prints that string straight
/// to stdout (it returns nothing): the content is meant to be *looked at* in a
/// terminal, not parsed — use the typed accessors (`entries`, `value`, …) for
/// programmatic access.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::dump::{self, Dump, DumpOptions};
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let mesh = Mesh::from_submesh(sm);
/// // `render` rend le texte, `dump` l'imprime : le contenu est fait pour
/// // être **regardé** dans un terminal, non analysé — les accesseurs
/// // typés sont là pour le reste.
/// let texte = mesh.render(&DumpOptions::default());
/// assert!(texte.contains("SEG2"));
/// // `dump_with` est la même chose, imprimée avec des options choisies.
/// mesh.dump_with(&DumpOptions { precision: 6, ..Default::default() });
/// ```
pub trait Dump {
    /// Render the full content as a `String`. This is the composition core
    /// (aggregates concatenate their sub-objects' renders); end users normally
    /// call [`dump`](Dump::dump) instead.
    fn render(&self, opts: &DumpOptions) -> String;

    /// Print the full content to stdout (default [`DumpOptions`]).
    fn dump(&self) {
        self.dump_with(&DumpOptions::default());
    }

    /// Print the full content to stdout, honouring the supplied options.
    fn dump_with(&self, opts: &DumpOptions) {
        println!("{}", self.render(opts));
    }
}

/// Print `text` through Python's built-in `print` (so it honours
/// `sys.stdout`, redirection, and is captured by test harnesses), then return.
///
/// Used by the `dump` pymethods, which print to the terminal rather than
/// returning a string.
#[cfg(feature = "python-api")]
pub fn py_print(py: pyo3::Python<'_>, text: &str) -> pyo3::PyResult<()> {
    use pyo3::types::PyAnyMethods;
    py.import("builtins")?.call_method1("print", (text,))?;
    Ok(())
}

// ─── Shared formatting helpers ──────────────────────────────────────────────

/// Format a float with `precision` digits after the point.
///
/// ```
/// # use pyrucast::dump::fmt_float;
/// // Précision fixe : les zéros de queue sont **conservés**, pour que les
/// // colonnes d'un tableau s'alignent.
/// assert_eq!(fmt_float(1.5, 3), "1.500");
/// assert_eq!(fmt_float(-0.25, 1), "-0.2");
/// ```
pub fn fmt_float(v: f64, precision: usize) -> String {
    format!("{:.*}", precision, v)
}

/// Render a right-aligned text table with row/column elision.
///
/// `headers` is the full header row; each entry of `rows` must have the same
/// length as `headers`. Column 0 is treated as a **label column** and is always
/// kept; the remaining columns are capped at `opts.max_cols`. Rows are capped at
/// `opts.max_rows`. Truncation appends `⋮` cue rows/columns and a trailing
/// `… (N de plus)` note.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::dump::{self, Dump, DumpOptions};
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let mesh = Mesh::from_submesh(sm);
/// // La colonne 0 est une **colonne d'étiquettes** : elle est toujours
/// // gardée, les autres étant plafonnées.
/// let entetes = vec!["nœud".to_string(), "x".to_string()];
/// let lignes = vec![vec!["0".to_string(), "0.000".to_string()]];
/// let t = dump::table(&entetes, &lignes, &DumpOptions::default());
/// assert!(t.contains("nœud") && t.contains("0.000"));
/// // Au-delà du plafond de lignes, l'élision est annoncée.
/// let longues: Vec<Vec<String>> = (0..50)
///     .map(|i| vec![i.to_string(), "0".to_string()]).collect();
/// assert!(dump::table(&entetes, &longues, &DumpOptions::default())
///     .contains("de plus"));
/// ```
pub fn table(headers: &[String], rows: &[Vec<String>], opts: &DumpOptions) -> String {
    let ncol = headers.len();
    if ncol == 0 {
        return String::new();
    }

    // Column selection: keep col 0 (labels), cap value columns at max_cols.
    let value_cols = ncol - 1;
    let shown_value_cols = value_cols.min(opts.max_cols);
    let col_truncated = value_cols > shown_value_cols;
    let shown_cols = 1 + shown_value_cols;

    // Row selection.
    let shown_rows = rows.len().min(opts.max_rows);
    let row_truncated = rows.len() > shown_rows;

    // Assemble the visible string grid (header + data + optional cue row).
    let mut grid: Vec<Vec<String>> = Vec::with_capacity(shown_rows + 2);
    let build_line = |src: &[String]| -> Vec<String> {
        let mut line: Vec<String> = src[..shown_cols].to_vec();
        if col_truncated {
            line.push("…".to_string());
        }
        line
    };
    grid.push(build_line(headers));
    for r in &rows[..shown_rows] {
        grid.push(build_line(r));
    }
    if row_truncated {
        let n = if col_truncated {
            shown_cols + 1
        } else {
            shown_cols
        };
        grid.push(vec!["⋮".to_string(); n]);
    }

    // Per-column widths (char count: our labels are ASCII + a few 1-char glyphs).
    let cols = grid[0].len();
    let mut widths = vec![0usize; cols];
    for line in &grid {
        for (i, cell) in line.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }

    // Render right-aligned, two-space column separator.
    let mut out = String::new();
    for line in &grid {
        for (i, cell) in line.iter().enumerate() {
            if i > 0 {
                out.push_str("  ");
            }
            for _ in 0..widths[i] - cell.chars().count() {
                out.push(' ');
            }
            out.push_str(cell);
        }
        out.push('\n');
    }

    if row_truncated {
        out.push_str(&format!(
            "… ({} ligne(s) de plus)\n",
            rows.len() - shown_rows
        ));
    }
    if col_truncated {
        out.push_str(&format!(
            "… ({} colonne(s) de plus)\n",
            value_cols - shown_value_cols
        ));
    }
    out
}

/// Render a labeled dense grid (`data` row-major, `row_labels.len()` ×
/// `col_labels.len()`) with in-line labels and elision.
///
/// Used for matrix dumps: row/column DOF labels sit directly on the grid.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::dump::{self, Dump, DumpOptions};
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let mesh = Mesh::from_submesh(sm);
/// // Une grille dense dont les étiquettes de ligne et de colonne sont
/// // posées **sur** la grille — ce dont vit le dump d'une matrice.
/// let l = vec!["a".to_string(), "b".to_string()];
/// let g = dump::labeled_grid(&l, &l, &[1.0, 0.0, 0.0, 1.0], &DumpOptions::default());
/// assert!(g.contains('a') && g.contains("1.000"));
/// ```
pub fn labeled_grid(
    row_labels: &[String],
    col_labels: &[String],
    data: &[f64],
    opts: &DumpOptions,
) -> String {
    let nc = col_labels.len();
    let mut headers = Vec::with_capacity(nc + 1);
    headers.push(String::new()); // empty top-left corner
    headers.extend(col_labels.iter().cloned());

    let rows: Vec<Vec<String>> = row_labels
        .iter()
        .enumerate()
        .map(|(i, rl)| {
            let mut line = Vec::with_capacity(nc + 1);
            line.push(rl.clone());
            for j in 0..nc {
                line.push(fmt_float(data[i * nc + j], opts.precision));
            }
            line
        })
        .collect();

    table(&headers, &rows, opts)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_labeled_grid_aligns() {
        let rows = vec!["(n1,q)".into(), "(n2,q)".into()];
        let cols = vec!["(n1,T)".into(), "(n2,T)".into()];
        let data = vec![2.0, -1.0, -1.0, 2.0];
        let s = labeled_grid(&rows, &cols, &data, &DumpOptions::default());
        let expected = concat!(
            "        (n1,T)  (n2,T)\n",
            "(n1,q)   2.000  -1.000\n",
            "(n2,q)  -1.000   2.000\n",
        );
        assert_eq!(s, expected);
    }

    #[test]
    fn row_elision_marks_overflow() {
        let headers = vec!["node".into(), "T".into()];
        let rows: Vec<Vec<String>> = (0..5)
            .map(|i| vec![i.to_string(), format!("{i}.0")])
            .collect();
        let opts = DumpOptions {
            precision: 1,
            max_rows: 2,
            max_cols: 12,
        };
        let s = table(&headers, &rows, &opts);
        assert!(s.contains("⋮"), "expected an elision cue row:\n{s}");
        assert!(
            s.contains("3 ligne(s) de plus"),
            "expected a row note:\n{s}"
        );
    }

    #[test]
    fn col_elision_keeps_label_column() {
        // 1 label column + 4 value columns, capped at 2.
        let headers: Vec<String> = std::iter::once("node".to_string())
            .chain((0..4).map(|c| format!("c{c}")))
            .collect();
        let rows = vec![vec![
            "n0".into(),
            "1".into(),
            "2".into(),
            "3".into(),
            "4".into(),
        ]];
        let opts = DumpOptions {
            precision: 1,
            max_rows: 20,
            max_cols: 2,
        };
        let s = table(&headers, &rows, &opts);
        assert!(s.contains("node"), "label column must survive:\n{s}");
        assert!(
            s.contains("2 colonne(s) de plus"),
            "expected a column note:\n{s}"
        );
    }

    #[test]
    fn default_render_uses_default_options() {
        struct One;
        impl Dump for One {
            fn render(&self, o: &DumpOptions) -> String {
                fmt_float(1.0, o.precision)
            }
        }
        assert_eq!(One.render(&DumpOptions::default()), "1.000");
    }
}
