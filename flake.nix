{
	description = "Python Flake Exploration Mechanism Analysis";

	inputs = {
		# better use commit
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
		devenv.url = "github:cachix/devenv/v1.3.1";
		nixpkgs-python.url = "github:cachix/nixpkgs-python";
        nixpkgs-python.inputs = { nixpkgs.follows = "nixpkgs"; };
	};

	outputs = {self, nixpkgs, devenv, nixpkgs-python} @ inputs:

        let system = "x86_64-linux";
        pkgs = nixpkgs.legacyPackages.${system};

		in {
		    packages.${system}.devenv-up = self.devShells.${system}.default.config.procfileScript;

	        devShells.${system}.default = devenv.lib.mkShell {
	            inherit inputs pkgs;
	            modules = [
	                ({ pkgs, config, ... }: {
	                packages = with pkgs; [stdenv.cc.cc.lib gcc-unwrapped libz];
	                env.PYTHONPATH = "${./.}";
	                env.LD_LIBRARY_PATH = "${pkgs.gcc-unwrapped.lib}/lib64:${pkgs.libz}/lib";
                    # This is your devenv configuration
                        languages.python = {
                            enable = true;
                            version = "3.11.9";
                            venv.enable = true;
                            venv.requirements = ./requirements.txt;
                        };
                        enterShell = ''
                            python --version
                            python -m pip list
                        '';
                    })
                ];
            };
	    };
}
