{
  description = "Build the environment for follow-up submission to milestone D2.5";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells."${system}".default = pkgs.mkShell {
        name = "D2.5";
        packages = with pkgs; [
          (python312.withPackages (
            ps: with ps; [
              jax
              flax
              numpy
              seaborn
            ]
          ))
        ];
      };
    };
}
