{
  description = "Build the environment for follow-up submission to milestone D2.5";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        configuration = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      python = pkgs.python312;
      utils = python.pkgs.buildPythonPackage {
        name = "utilities";
        version = "0.1.0";
        src = ./src;
        buildInputs = with python.pkgs; [
          setuptools
          wheel
        ];
        pyproject = true;
      };
      python-with-packages = python.withPackages (
        p: with p; [
          utils
          jax
          numpy
          seaborn
          ipykernel
          chex
        ]
      );
    in
    {
      devShells."${system}".default = pkgs.mkShell {
        name = "D2.5";
        buildInputs = [
          python-with-packages
        ];
      };
    };
}
