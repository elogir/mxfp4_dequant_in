# MXFP4 Dequantizer

Loader de safetensors et dequantizer de tenseurs MXFP4, en Zig.

Les fonctions sont censées respecter la [spec OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

## Build & Run

Nécessite Zig 0.15.2 minimum.

```shell
# Build le projet
zig build

# Run le projet
zig build run -- input.safetensors output.safetensors
```

## Tester le code

Le modèle de test est `tiny_gpt_oss`, une version allégée de `gpt_oss`, pour tester rapidement le fonctionnement et comparer avec une version pré-dequantizée (notre ref).

Un utilitaire gentiment généré par un LLM est à disposition :

```shell
./test.sh
```

Celui-ci :
1. Crée un venv python pour pouvoir plus tard comparer les résultats de la ref à notre programme
2. Execute notre code sur le modèle de test `tiny_gpt_oss`
3. Compare les tenseurs un par un avec le nom (puisque l'ordre a été changé)

## Format MXFP4

MXFP4 stocke des floats sur 4 bits (1 signe, 2 exposant, 1 mantisse). Les tenseurs sont organisés en blocks de 32 éléments, chaque block ayant un facteur d'échelle E8M0 (exposant 8-bit, pas de mantisse).

Pour dequantizer :
1. On lit le scale E8M0 du block -> conversion en float32 -> splat
2. Pour chaque élément FP4 du block :
   - Conversion FP4 -> float32 via LUT (calculée à compile-time)
   - Multiplication par la scale (SIMD)

## La logique du Reader

1. Les headers sont parsés dans le init de notre Reader, cela permet de pouvoir directement les retourner au premier call.
2. Ensuite à chaque call, on va read un tenseur et là deux cas se présentent :
    * Si c'est un tenseur non-quantizé, on le copie uniquement
    * Si c'est une paire de tenseurs quantizés, on le dequant

    Ensuite on le garde jusqu'à ce que le caller demande tout le contenu.
3. Fin

## Notes

Ma plus grosse erreur a sûrement été de commencer avec le code sans exposer un `std.Io.Reader` (voir branch non-reader) puis une fois le tout terminé, tout retransformer en `std.Io.Reader`. Comprendre comment fonctionne le nouvel Io est déjà compliqué et manque un peu de guides, mais c'est possible de s'en sortir avec la doc et les sources de la librairie standard (et [la vidéo de ComputerBread](https://youtu.be/k74veXOMf4U?si=mGi4MI5Gy8DgRTUQ)). Peut-être qu'en partant directement sur l'approche attendue, cela aurait été plus simple et surtout plus optimal tout en pouvant "override" un max les fonctions de `std.Io.Reader`, où du coup, j'utilise l'implémentation de base.

Ma deuxième erreur aura été mon idée de vouloir parser les tenseurs dans le header et dans la data en supportant le fait que les tenseurs de blocks et de scales ne soient pas adjacents, m'obligeant à parser avec une hashmap pour faire correspondre les paires (plus expliqué dans le code) ce qui a bien entendu rendu la tâche plus difficile.

J'ai bien testé sur un safetensors complet de `gpt_oss_20b`, la dequant est successful mais je n'ai pas pu comparer les résultats ni [ouvrir le safetensors pour le parser](https://github.com/by321/safetensors_util), ma machine n'ayant pas assez de RAM... Cependant la taille parait correcte.

Cela étant l'un de mes premiers projets Zig, la structure, le flow et les allocations peuvent incontestablement être améliorés, mais j'aurais d'ici là l'occasion de plus pratiquer.

🎉

